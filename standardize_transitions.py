"""
Standardize all ground truth transition files to consistent format
- Converts mm:ss to seconds where needed
- Removes comments and empty lines
- Handles pipe format (keeps only transition time)
- Fixes file naming inconsistencies
"""

import re
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_timestamp(timestamp_str: str) -> float:
    """
    Parse timestamp from various formats to seconds
    
    Supports:
    - Seconds: "1.41" → 1.41
    - Minutes:Seconds: "1:41" → 101
    - mm:ss: "3:44" → 224
    - Pipe format: "1.41|1.40" → 1.41 (ignores ideal frame)
    """
    timestamp_str = timestamp_str.strip()
    
    if not timestamp_str or timestamp_str.startswith('#'):
        return None
    
    # Handle pipe format (keep only first part)
    if '|' in timestamp_str:
        timestamp_str = timestamp_str.split('|')[0].strip()
    
    # Try mm.ss format where '.' separates minutes and seconds (e.g., 3.44 => 3m 44s)
    dot_match = re.fullmatch(r"\s*(\d+)\.(\d{2})\s*", timestamp_str)
    if dot_match:
        minutes = int(dot_match.group(1))
        seconds = int(dot_match.group(2))
        if seconds < 60:
            return minutes * 60 + seconds
        # Fall through if seconds part invalid

    # Try simple seconds format
    try:
        return float(timestamp_str)
    except ValueError:
        pass
    
    # Try mm:ss format
    if ':' in timestamp_str:
        try:
            parts = timestamp_str.split(':')
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        except (ValueError, IndexError):
            logger.warning(f"Could not parse: {timestamp_str}")
            return None
    
    logger.warning(f"Unknown format: {timestamp_str}")
    return None


def standardize_file(file_path: Path) -> bool:
    """Standardize a single transition file"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        timestamps = []
        for line in lines:
            ts = parse_timestamp(line)
            if ts is not None:
                timestamps.append(ts)
        
        # Sort by timestamp
        timestamps.sort()
        
        # Write standardized format
        with open(file_path, 'w') as f:
            f.write(f"# Detected transitions (standardized format)\n")
            f.write(f"# Format: timestamp in seconds\n")
            f.write(f"# Total: {len(timestamps)} transitions\n\n")
            
            for ts in timestamps:
                f.write(f"{ts:.2f}\n")
        
        logger.info(f"✓ Standardized: {file_path.name} ({len(timestamps)} transitions)")
        return True
    
    except Exception as e:
        logger.error(f"✗ Error standardizing {file_path.name}: {e}")
        return False


def find_all_transition_files(root_dir: Path):
    """Find all transition files in directory tree"""
    patterns = ['*_transitions.txt', '*_transtions.txt', 'transitions.txt']
    files = []
    
    for pattern in patterns:
        files.extend(root_dir.rglob(pattern))
    
    return sorted(set(files))  # Remove duplicates


def fix_filename_typos(root_dir: Path):
    """Fix common typos like 'transtions' → 'transitions'"""
    for file_path in root_dir.rglob('*_transtions.txt'):
        new_name = str(file_path).replace('_transtions.txt', '_transitions.txt')
        new_path = Path(new_name)
        
        if not new_path.exists():
            try:
                file_path.rename(new_path)
                logger.info(f"✓ Renamed: {file_path.name} → {new_path.name}")
            except Exception as e:
                logger.error(f"✗ Could not rename {file_path.name}: {e}")
        else:
            logger.warning(f"⚠ Target file already exists: {new_path.name}")


def main():
    """Main standardization routine"""
    root = Path('data')
    
    logger.info(f"\n{'='*70}")
    logger.info(f"STANDARDIZING TRANSITION FILES")
    logger.info(f"{'='*70}\n")
    
    # Fix typos first
    logger.info("Step 1: Fixing filename typos...")
    fix_filename_typos(root)
    
    # Find all transition files
    logger.info("\nStep 2: Finding all transition files...")
    transition_files = find_all_transition_files(root)
    logger.info(f"Found {len(transition_files)} transition file(s)\n")
    
    if not transition_files:
        logger.error("No transition files found!")
        return
    
    # Standardize each file
    logger.info("Step 3: Standardizing format...")
    success_count = 0
    for file_path in transition_files:
        if standardize_file(file_path):
            success_count += 1
    
    logger.info(f"\n{'='*70}")
    logger.info(f"COMPLETE: {success_count}/{len(transition_files)} files standardized")
    logger.info(f"{'='*70}\n")
    
    logger.info("All transition files now use consistent format:")
    logger.info("  Format: timestamp in seconds (one per line)")
    logger.info("  Example:")
    logger.info("    1.41")
    logger.info("    2.31")
    logger.info("    4.38")


if __name__ == "__main__":
    main()
