"""Select best slide screenshots before transitions (full content, not occluded)."""
import pandas as pd
from pathlib import Path


class SlideSelector:
    """Identify and rank best pre-transition frames for slide extraction."""
    
    def __init__(self, occlusion_threshold=0.5, min_content_fullness=0.7, min_frame_quality=0.3):
        """
        Args:
            occlusion_threshold: max is_occluded value to accept (0=strict, 1=allow all)
            min_content_fullness: minimum content ratio to consider frame "full"
            min_frame_quality: minimum quality score (sharpness+contrast)
        """
        self.occlusion_threshold = occlusion_threshold
        self.min_content_fullness = min_content_fullness
        self.min_frame_quality = min_frame_quality
    
    def select_best_slides(self, frames_df, lookback_window=10, top_n_per_transition=5):
        """
        For each transition, pick multiple best frames within lookback_window before it.
        
        Args:
            frames_df: DataFrame with columns: video_name (or video_id), timestamp, is_transition, 
                       is_occluded, content_fullness, frame_quality, frame_path
            lookback_window: number of frames to look back before transition
            top_n_per_transition: how many best frames to keep per transition (default 5)
        
        Returns:
            DataFrame with best slide frames only
        """
        best_slides = []
        
        # Handle both video_name and video_id column names
        vid_col = 'video_id' if 'video_id' in frames_df.columns else 'video_name'
        
        for video_id, video_group in frames_df.groupby(vid_col):
            video_group = video_group.sort_values('timestamp').reset_index(drop=True)
            
            # Find transition frames
            transition_indices = video_group[video_group['is_transition'] == True].index.tolist()
            
            for trans_idx in transition_indices:
                # Look back N frames before transition
                start_idx = max(0, trans_idx - lookback_window)
                candidates = video_group.loc[start_idx:trans_idx-1].copy() if trans_idx > 0 else pd.DataFrame()
                
                if candidates.empty:
                    continue
                
                # Score all candidates (weighted: fullness + quality - occlusion penalty)
                candidates['score'] = (
                    0.5 * candidates['content_fullness'] + 
                    0.4 * candidates['frame_quality'] -
                    0.3 * candidates['is_occluded']  # penalize occlusion
                )
                
                # Sort by score and take top N
                top_candidates = candidates.nlargest(min(top_n_per_transition, len(candidates)), 'score')
                
                # Add transition_id to track which transition these belong to
                for idx, row in top_candidates.iterrows():
                    row_dict = row.to_dict()
                    row_dict['transition_id'] = f"{video_id}_T{len([x for x in best_slides if x.get(vid_col)==video_id]) // top_n_per_transition + 1}"
                    row_dict['rank'] = top_candidates.index.get_loc(idx) + 1
                    best_slides.append(row_dict)
        
        return pd.DataFrame(best_slides) if best_slides else pd.DataFrame()
    
    def save_slide_manifest(self, best_slides_df, output_path):
        """Save selected slides to CSV with clean schema."""
        if best_slides_df.empty:
            print(f"No slides selected; skipping save to {output_path}")
            return
        
        # Reorder and clean columns
        output_cols = [
            'video_name', 'transition_id', 'rank', 'timestamp', 'frame_path',
            'is_occluded', 'skin_ratio', 'content_fullness', 'frame_quality', 
            'board_type', 'score'
        ]
        
        # Handle both video_name and video_id
        if 'video_id' in best_slides_df.columns and 'video_name' not in best_slides_df.columns:
            output_cols[0] = 'video_id'
        
        available_cols = [c for c in output_cols if c in best_slides_df.columns]
        clean_df = best_slides_df[available_cols].copy()
        
        clean_df.to_csv(output_path, index=False)
        print(f"Saved {len(clean_df)} best slide frames -> {output_path}")
        
        return clean_df
