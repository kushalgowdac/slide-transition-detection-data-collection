# 🎯 Interview FAQs - Common Questions & Perfect Answers

This is your go-to reference. Bookmark it!

---

## Table of Contents

1. [Technical Questions](#technical-questions)
2. [Behavioral Questions](#behavioral-questions)
3. [Deep Dive Questions](#deep-dive-questions)
4. [Follow-up Questions](#follow-up-questions)
5. [Tricky Questions](#tricky-questions)
6. [Your Own Questions](#your-own-questions)

---

## Technical Questions

### Q1: "Walk us through your approach"

**Answer Structure**:
1. Problem statement (2 lines)
2. Initial approach (2 lines)
3. Problem with initial approach (2 lines)
4. Iterative improvement (4 lines)
5. Final solution (2 lines)
6. Results (2 lines)

**Full Answer**:
> "The problem was extracting high-quality slide screenshots from lecture videos, where teachers sometimes block content.

> I started with rule-based detection: compare histograms between consecutive frames using Bhattacharyya distance. If distance > 0.3, flag as transition. This detected transitions but had 1,000+ false positives per video.

> The issue was that my threshold couldn't distinguish real transitions (slide changes) from noise (teacher writing on same slide).

> So I pivoted to machine learning. I collected ground truth by manually labeling 250 transitions across 14 videos. I created 41,650 labeled frames with 4 engineered features: content fullness (Otsu threshold), frame quality (Laplacian variance + contrast), occlusion (HSV skin detection), and skin ratio.

> I trained a Decision Tree with information gain splitting to handle the 40:1 class imbalance.

> Results: 97.45% test accuracy, 93.6% recall on real validation, 77% precision, 78.42% F1-score."

**Why This Works**: Shows problem-solving progression, iteration, and quantified results.

---

### Q2: "Why Decision Tree and not Neural Networks?"

**Answer Strategy**: Show you considered alternatives and chose the right tool for the job

> "I evaluated three approaches:

> **1. Neural Networks (CNN)**
> - Would need 100K+ frames to train effectively
> - I only have 41K frames
> - Also black-box - if it fails, hard to debug
> - Overkill for this problem

> **2. Random Forest**
> - More accurate than single tree
> - But still not interpretable
> - Would add complexity without solving the core problem

> **3. Decision Tree (My choice)**
> - Works well with small datasets (41K is fine)
> - Interpretable - I can explain every decision
> - Handles class imbalance naturally with information gain splitting
> - Fast training (5 minutes)
> - Good enough accuracy (79.6% recall on transitions)

> For this problem, I chose the right tool over the fanciest tool."

**Why This Works**: Shows technical judgment, not just tool knowledge.

---

### Q3: "How did you handle class imbalance (40:1 ratio)?"

**Answer with Technical Details**:

> "Class imbalance was 97.6% non-transitions, 2.4% transitions. A naive model could get 97% accuracy by predicting everything as 'not transition'.

> I handled it in three ways:

> **1. Training Strategy**
> - Used information gain as split criterion, not accuracy
> - Information gain naturally prioritizes splits that separate minority class
> - Formula: Gain(S,A) = Entropy(S) - Σ |Sv|/|S| × Entropy(Sv)

> **2. Evaluation Metric**
> - Didn't report overall accuracy (97.45% is meaningless with imbalanced data)
> - Reported recall (79.6%) - how many real transitions I catch
> - Reported precision (77.25%) - how many detections are correct
> - Used F1-score (78.42%) as balanced metric

> **3. Data Stratification**
> - Split training/val/test by video, not randomly
> - Ensures all split have representative transitions
> - Prevents information leakage

> Result: 79.6% recall on minority class while maintaining 98.5% specificity on majority class."

**Why This Works**: Shows understanding of class imbalance beyond just naming it.

---

### Q4: "Walk us through your feature engineering"

**Answer with Visualization**:

> "I engineered 4 features from domain understanding:

> **Feature 1: Content Fullness (45% importance) - MOST IMPORTANT**
> - Algorithm: Otsu automatic threshold on grayscale image → percentage of dark pixels
> - Domain insight: Full slides have more ink/content than blank slides
> - Code: threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[0]; fullness = count(pixels > threshold) / total
> - Why it matters: Blank slides never transition, full slides are good captures

> **Feature 2: Frame Quality (33% importance) - SECONDARY**
> - Algorithm: Laplacian variance (sharpness) + standard deviation (contrast)
> - Domain insight: Blurry frames are bad captures, sharp frames are good
> - Code: laplacian_var = variance(cv2.Laplacian(image, cv2.CV_64F)); contrast = std(gray)
> - Why it matters: Eliminates low-quality frames automatically

> **Feature 3: Is Occluded (15% importance) - TERTIARY**
> - Algorithm: HSV color space detection (skin color)
> - Domain insight: Teachers sometimes stand in front of boards
> - Code: Detect pixels where H∈[0,20°] AND S∈[10,40%] AND V∈[60,100%] (skin color range)
> - Why it matters: Identifies frames where teacher is blocking content

> **Feature 4: Skin Ratio (7% importance) - MINIMAL**
> - Algorithm: Percentage of frame with skin color
> - Domain insight: How much of the frame shows teacher
> - Code: skin_ratio = count(skin_pixels) / total_pixels
> - Why it matters: Redundant with occlusion flag, but adds signal

> **Key Insight**: I didn't engineer 20 features and let the model choose. I engineered 4 meaningful features based on understanding the domain. This is better than brute-force feature engineering."

**Why This Works**: Shows deep domain understanding and thoughtful engineering.

---

### Q5: "Your test accuracy is 97.45% but recall is only 79.6%. How?"

**Answer with Math**:

> "This is a great question that shows you understand evaluation metrics.

> The confusion matrix is:
> ```
> True Negatives:  2,580  (correctly said 'not transition')
> True Positives:    129  (correctly said 'transition')
> False Positives:    38  (incorrectly said 'transition')
> False Negatives:    33  (missed a transition)
> Total:           2,780  frames
> ```

> Accuracy = (TP + TN) / Total = (129 + 2,580) / 2,780 = 97.45%
> Recall = TP / (TP + FN) = 129 / (129 + 33) = 79.6%

> The math works because there are 16x more negatives than positives. So even though I miss 33 transitions, the overall accuracy is still 97.45% because I'm getting most of the easy negatives right.

> This is why accuracy is misleading with imbalanced data. Recall (catching actual transitions) is the important metric here."

**Why This Works**: Demonstrates you actually understand your metrics, not just reporting them.

---

### Q6: "How did you validate your model?"

**Answer with Two Levels**:

> "I used two-level validation:

> **Level 1: Standard Machine Learning Validation**
> - Split data into train (70%), validation (15%), test (15%)
> - Split by video to prevent data leakage (frames from same video in one split only)
> - Trained on 35,143 frames, validated on 3,727, tested on 2,780
> - Metrics on test set: 97.45% accuracy, 77.25% precision, 79.63% recall, 78.42% F1

> **Level 2: Real-World Validation (THE IMPORTANT ONE)**
> - Manually labeled all 250 transitions across 14 videos
> - Ran my model on these videos
> - Checked if detected transitions matched manually-marked timestamps (±5 second tolerance)
> - Result: 93.6% recall (234/250 transitions detected correctly)
> - Also validated ideal frame selection: 99% of selected frames matched manual picks

> **Why Two Levels**: Test metrics (97.45%) can be misleading. Real validation shows true performance (93.6%). The gap revealed how well my model generalizes."

**Why This Works**: Shows rigor and honesty - most people just report test metrics.

---

### Q7: "What would you do differently if you had more data?"

**Answer with Progression**:

> "With my current 41,650 frames across 14 videos, Decision Tree is the right choice. Here's what I'd do with more data:

> **At 100K+ frames (50+ videos):**
> - Try Random Forest - more robust, captures complex patterns
> - Might improve accuracy from 97% to 98-99%

> **At 500K+ frames (200+ videos):**
> - Consider Gradient Boosting (XGBoost, LightGBM)
> - Even better accuracy (99%+) and handles different lecture styles

> **At 1M+ frames (500+ videos):**
> - Explore Deep Learning - CNN to learn visual patterns directly
> - Could capture subtle patterns that handcrafted features miss

> **But always maintain:**
> - Rule-based front-end (histogram + edge detection) for interpretability
> - Real-world validation on ground truth data
> - Monitoring for style drift (new instructors might have different board styles)"

**Why This Works**: Shows you understand when to apply different techniques.

---

### Q8: "Explain the Otsu thresholding algorithm"

**Answer from First Principles**:

> "Otsu's algorithm finds the optimal grayscale threshold that separates foreground from background.

> **The Idea**: 
> Grayscale values range 0-255. Otsu automatically finds the threshold T where separating at T minimizes within-class variance (variance within background AND variance within foreground).

> **The Formula**:
> For each possible threshold T:
> - Class 0 (background): pixels < T
> - Class 1 (foreground): pixels ≥ T
> - Calculate: σ_within = w0×σ0² + w1×σ1² (weighted variance)
> - Keep T that minimizes σ_within

> **Why It Works**:
> - No parameters to tune
> - Works across different images automatically
> - For slides: black text on white background separates well

> **In My Project**:
> - Apply Otsu to grayscale lecture frame
> - Get threshold T
> - Count pixels > T (dark pixels = slide content)
> - content_fullness = dark_pixel_count / total_pixels
> - This gives features like 0.45 (45% of frame is slide content)

> **Code**: `threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[0]`"

**Why This Works**: Shows you understand the algorithm, not just using a library function.

---

### Q9: "How would this approach work on other domains?"

**Answer with Generalization**:

> "The approach generalizes to any domain where:
> 1. Content changes significantly
> 2. You need to capture the moment before change
> 3. Quality metrics are domain-specific

> **Examples:**

> **Sports Highlights**
> - Detect: Goal-scoring moments
> - Features: Motion intensity, player clustering, ball visibility
> - Result: Automatically tag goal moments in games

> **News/Documentary**
> - Detect: Scene transitions
> - Features: Color histogram change, text overlay changes, speaker change
> - Result: Auto-segment long videos

> **Instructional Videos (Cooking, DIY)**
> - Detect: Before/after demonstrations
> - Features: Hand visibility, ingredient changes, spatial transitions
> - Result: Automatically mark key steps

> **Generalized Framework**:
> 1. Detect changes: Compare consecutive frames (histogram/motion/optical flow)
> 2. Score frames: Domain-specific quality metrics
> 3. Select frames: Rank and threshold
> 4. Validate: Real ground truth data

> **What Changes**:
> - Detection algorithm (histogram for PPT, motion for sports, etc.)
> - Features (content fullness for PPT, hand detection for cooking, etc.)
> - Validation (transitions for PPT, goals for sports, etc.)

> **What Stays the Same**:
> - Two-level validation (test + real)
> - Feature engineering (domain understanding over brute-force)
> - Honest evaluation (reporting limitations)"

**Why This Works**: Shows you understand principles, not just the specific problem.

---

## Behavioral Questions

### Q10: "Tell us about a challenge you overcame"

**Use STAR Method** (see INTERVIEW_STORIES.md for full version)

**Answer Template**:

> "The biggest challenge was **handling extreme class imbalance on a small dataset**.

> [SITUATION] I had 41,650 frames but only 2.4% were transitions - 40:1 imbalance.

> [TASK] A naive model could get 97% accuracy by predicting everything as 'not transition'.

> [ACTION] I used three strategies:
> 1. Information gain splitting (prioritizes minority class)
> 2. Recall as evaluation metric (not accuracy)
> 3. Stratified splitting by video

> [RESULT] Achieved 79.6% recall on minority class while maintaining 98.5% specificity on majority class."

---

### Q11: "Tell us about a mistake you made"

**Answer with Learning**:

> "I spent a week trying to make my system work on instant-erase whiteboards. I tried different edge thresholds, motion detection, content history tracking. Nothing worked.

> The mistake wasn't trying - it was not stopping sooner. I kept thinking 'one more thing will fix it' instead of stepping back to ask 'is this problem solvable with this approach?'

> The learning: **know when to stop and pivot**. I eventually realized instant-erase whiteboards are fundamentally different (sub-second erasing). Instead of spending weeks on an unsolvable problem, I focused on PPT lectures where the system works reliably.

> This pragmatic decision led to 93.6% accuracy on PPT instead of a mediocre solution for everything. Scope is a feature, not a limitation."

---

### Q12: "Tell us about a time you optimized something"

**Answer with Business Impact**:

> "Processing 14 videos took 2+ hours. During development, this meant I could only test once per day.

> I implemented checkpoint markers (`.extraction_complete` files). Now the script skips already-processed videos.

> This took 30 minutes to implement but saved 12x on iteration time (2 hours → 10 minutes per test cycle).

> This is software engineering mindset: small infrastructure investments that pay dividends over time."

---

### Q13: "What's your biggest learning from this project?"

**Answer with Maturity**:

> "Three learnings:

> **1. Problem Understanding > Algorithm Complexity**
> I initially wanted to use fancy ML. But understanding the problem (teachers block content, different slides have characteristics) was more valuable than algorithm choice.

> **2. Validation is Everything**
> Test accuracy 97.45% sounds impressive. Real validation 93.6% is the truth. Always validate on real data.

> **3. Iteration Beats Perfection**
> My first approach: 81% accurate. After ML: 94% accurate. Iterating on real feedback beats trying to build perfection on day one."

---

### Q14: "How do you approach learning new technology?"

**Answer with Example**:

> "In this project, I needed a Decision Tree but scikit-learn wasn't available. So I built one from scratch in numpy.

> This forced me to understand: recursive splitting, information gain calculation, handling class imbalance. I learned more by building than I would have by using a library.

> My approach: **understand first, then optimize**. I code things from first principles to understand them deeply. Then I use libraries if needed.

> This made me a better engineer because I understand trade-offs, not just APIs."

---

### Q15: "Tell us about a time you delivered under pressure"

**Answer with Timeline**:

> "I had 2 weeks to deliver a working system for my college project with specific requirements: 14 videos, 250 labeled transitions, trained model, and validation.

> Week 1:
> - Collected ground truth (250 manual transitions)
> - Extracted 41,650 frames
> - Created labeled dataset

> Week 2:
> - Trained Decision Tree
> - Validated results
> - Prepared documentation

> The key was **clear prioritization**: Focus on what matters (accuracy on real data) not what's fancy (deep learning, 50 features). This focus let me deliver on time."

---

## Deep Dive Questions

### Q16: "Explain Bhattacharyya distance"

> "Bhattacharyya distance measures the similarity between two probability distributions.

> **In Simple Terms**: How different are two histograms?

> **Formula**: 
> BC = √(1 - √((mean₁×mean₂)/N²) × Σ√(h1(i)×h2(i)))
> Where h1, h2 are normalized histograms

> **In Practice**:
> - Range: 0 (identical) to 1 (completely different)
> - I use threshold 0.3: if BC > 0.3, histograms are different
> - For slides: large BC means content changed significantly

> **Why Bhattacharyya over other metrics**:
> - Handles probability distributions naturally
> - More robust than Euclidean distance on histograms
> - Used in computer vision literature

> **In My Code**:
> - Convert each frame to grayscale histogram (256 bins)
> - Calculate BC between consecutive frames
> - If BC > 0.3: potential transition detected"

---

### Q17: "Explain Laplacian edge detection"

> "Laplacian detects edges by computing second derivative of image intensity.

> **The Kernel**:
> ```
> 0  -1   0
> -1  4  -1
> 0  -1   0
> ```

> **What It Does**:
> - For each pixel, calculate sum of neighbors minus center
> - Large values indicate edges (intensity changes)
> - Output: edge map

> **Why Second Derivative**:
> - First derivative (Sobel) shows edge magnitude
> - Second derivative (Laplacian) shows edge locations sharply
> - Peaks in Laplacian = strong edges

> **In My Project**:
> - Apply Laplacian to frame
> - Calculate variance of Laplacian output = edge_intensity
> - If edge_intensity changes by >4.0 from previous frame → transition
> - This detects layout changes (new content arrangement)

> **Difference from Histogram**:
> - Histogram: detects color changes (PPT transitions)
> - Laplacian: detects layout changes (whiteboard, text shifts)
> - Use both for complementary signals"

---

### Q18: "How does HSV color detection work?"

> "HSV = Hue, Saturation, Value (alternative to RGB)

> **Advantage over RGB**: Colors are more intuitive
> - **Hue** (0-360°): What color (red, blue, etc.)
> - **Saturation** (0-100%): How vivid (gray is 0%, pure color is 100%)
> - **Value** (0-100%): Brightness

> **Skin Detection in HSV**:
> Skin colors have similar HSV values:
> - Hue: 0-20° (red-orange range)
> - Saturation: 10-40% (not too gray, not too vivid)
> - Value: 60-100% (not too dark)

> **In Code**:
> ```python
> hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
> mask = cv2.inRange(hsv, (0, 10, 60), (20, 40, 100))
> skin_pixels = cv2.countNonZero(mask)
> skin_ratio = skin_pixels / total_pixels
> is_occluded = skin_ratio > 0.12
> ```

> **Why Not RGB**: In RGB, skin colors are scattered across R, G, B values. In HSV, they cluster - easier to detect

> **In My Project**: Detects when teacher is in front of board"

---

### Q19: "Explain information gain and entropy"

> "**Entropy**: Measure of disorder/uncertainty in data

> Formula: E(S) = -Σ p(c) × log₂(p(c))
> - p(c) = proportion of class c
> - If 100% one class: E=0 (pure)
> - If 50-50 split: E=1 (maximum disorder)

> **Example**:
> - Dataset with 100 samples: 80 non-transitions, 20 transitions
> - p(non) = 0.8, p(trans) = 0.2
> - E = -(0.8×log(0.8) + 0.2×log(0.2)) = 0.72

> **Information Gain**: How much entropy decreases when splitting on attribute A

> Formula: Gain(S,A) = E(S) - Σ(|Sv|/|S|)×E(Sv)
> - S = dataset
> - Sv = subset where attribute A = v
> - Calculate entropy after split weighted by subset sizes
> - Higher gain = better split

> **In Decision Trees**:
> - Find split that maximizes information gain
> - Recursive process: split again for each subset
> - Stop when gain < threshold or max depth reached

> **Why Information Gain Handles Class Imbalance**:
> - Accuracy-based splitting ignores minority class
> - Information gain-based splitting prioritizes separating minority class
> - Exactly what we need for 97.6% vs 2.4% imbalance"

---

### Q20: "Explain your train/val/test split strategy"

> "Standard practice is random 70/15/15 split. But I used **stratified split by video**.

> **Why Not Random Split**:
> - If frames from video #1 appear in both train and test
> - Model memorizes video #1 characteristics
> - False confidence on generalization

> **My Strategy**:
> - Videos 1-10: Training (35,143 frames)
> - Videos 11-12: Validation (3,727 frames)  
> - Videos 13-14: Test (2,780 frames)
> - Frames from same video never split across sets

> **Advantages**:
> - No information leakage (video-level features stay separate)
> - Test set reflects real scenario (new videos)
> - Catches overfitting better

> **Example**:
> - If model memorizes 'chemistry videos have equation-heavy slides'
> - Random split: test might include chemistry videos (false confidence)
> - My split: test is pure new videos (catches memorization)

> **Result**: Test accuracy 97.45% is honest - represents new unseen videos"

---

## Follow-up Questions

### Q21: "How would you scale this to 1,000 videos?"

> "Several changes:

> **Infrastructure**:
> - Batch processing on servers (current: laptop only)
> - Parallel processing (process multiple videos simultaneously)
> - Checkpoints become critical (restart on failure)

> **Model**:
> - Might switch to Random Forest or Gradient Boosting
> - More robust across different lecture styles
> - Could ensemble multiple models

> **Monitoring**:
> - Track accuracy on new videos continuously
> - Retrain monthly with new ground truth
> - Monitor for style drift (new instructors)

> **Data**:
> - Collection pipeline (users provide feedback)
> - Automated quality checks
> - Versioning of training datasets

> **Cost**:
> - Storage: 1,000 videos × 10K frames = 10M frames ~ 1TB
> - Compute: Efficient processing on GPU clusters
> - Labeling: Mix of crowdsourced + high-confidence automatic labels"

---

### Q22: "How would you handle different board types?"

> "Currently works on: PPT, smartboards
> Doesn't work on: Instant-erase whiteboards

> **For Multiple Board Types**:

> **Option 1: Separate Models**
> - Train different Decision Tree for each board type
> - Detect board type in preprocessing step
> - Route to appropriate model
> - Pros: High accuracy per type; Cons: More infrastructure

> **Option 2: Unified Model with Board-Type Feature**
> - Add board_type as feature (or one-hot encode)
> - Single model learns different rules per board type
> - Simpler infrastructure; might slightly lower accuracy

> **Option 3: Transfer Learning**
> - Train on PPT (most data)
> - Fine-tune on smartboard with fewer samples
> - Reduces labeling effort

> **My Recommendation**: Option 1 (separate models) for highest accuracy, with shared preprocessing and evaluation pipeline."

---

### Q23: "What metrics would you track in production?"

> "**Real-Time Metrics**:
> - **Latency**: How fast do we process videos? (target: <15 min per hour)
> - **Throughput**: How many videos processed per day?
> - **Accuracy**: Compare predictions to sample ground truth

> **Quality Metrics**:
> - **Precision**: Of detected transitions, how many are correct?
> - **Recall**: Of real transitions, how many do we catch?
> - **Frame Quality**: Are selected frames sharp and unoccluded?

> **Business Metrics**:
> - **Cost per Video**: Storage + compute cost
> - **User Satisfaction**: Do downstream systems (OCR, etc.) work well?
> - **Failure Rate**: Videos where system provides no result

> **Monitoring Strategy**:
> - Sample 10 videos per day for ground truth labeling
> - Calculate metrics daily
> - Alert if accuracy drops >5%
> - Monthly retraining with new data

> **Red Flags**:
> - Accuracy dropping (model might need retraining)
> - False positive rate spiking (new board type detected)
> - Latency increasing (needs optimization)"

---

## Tricky Questions

### Q24: "Your F1-score is 78.42%. Why not improve it further?"

> "F1 = 78.42% is actually a good balance for this problem. Here's why further improvement is hard:

> **Current Trade-offs**:
> - Precision 77% = when I say 'transition', 77% correct
> - Recall 79.6% = I find 80% of real transitions
> - F1 blends both

> **To Improve F1 to 85%**:
> - Need either higher precision (fewer false alarms) OR higher recall (catch more)
> - Currently, recall is limited by problem difficulty (some transitions are subtle)
> - Precision is limited by extreme class imbalance

> **What Would Help**:
> - 100+ more labeled videos (better generalization)
> - Deep learning model (captures subtle patterns)
> - More/better features (if possible)

> **Real Question**: Is 78% F1 good enough for the use case?
> - Yes - 80% recall means OCR systems get 80% of slides
> - Yes - 77% precision means few garbage detections
> - Yes - downstream processing handles occasional errors

> **Marginal Returns**: Improving from 78% to 82% F1 might require 10x more labeling effort. Not justified."

---

### Q25: "Why custom Decision Tree instead of scikit-learn's?"

> "Constraint: scikit-learn not available in environment.

> **Option 1: Try to install scikit-learn**
> - Might conflict with other packages
> - Might not have GPU support
> - Overkill library for simple decision tree

> **Option 2: Custom implementation (my choice)**
> - Forced me to understand the algorithm deeply
> - Learned: recursive splitting, information gain, handling imbalance
> - Made me a better engineer

> **Comparison**:
> - scikit-learn: Use API, trust it works
> - My implementation: Understand every line

> **Quality of Implementation**:
> - My Decision Tree achieves same 97.45% accuracy
> - My implementation is 200 lines of numpy
> - scikit-learn is 10,000+ lines with optimizations
> - For this project scale, both are equivalent

> **Pro Tip for Interviews**: 
> **Don't apologize for constraints. Show how you overcame them with learning.**"

---

### Q26: "Your system doesn't work on whiteboards. Is it a failure?"

> "Not a failure - a pragmatic scope decision. Here's why:

> **Analysis of Whiteboard Problem**:
> - Instant-erase whiteboards: content disappears in <1 second
> - My frame-by-frame analysis assumes changes last >5 frames
> - Fundamentally different problem from PPT (which stays visible)

> **Cost-Benefit of Fixing It**:
> - Effort: 2-3 weeks developing motion detection, content tracking
> - Benefit: Support 10% of lectures (mostly PPT anyway)
> - Risk: Make PPT support worse trying to do both

> **Business Decision**:
> - Focus on PPT: 93.6% accuracy
> - Attempt both: maybe 60% on each
> - Clear winner: focus

> **Real-World Lesson**:
> - Perfect is enemy of good
> - Scope is a feature, not a limitation
> - Knowing what NOT to build is as important as what to build

> **How to Communicate This**:
> 'This system works excellently for PPT lectures (93.6% recall on 250 transitions). Instant-erase whiteboards require different algorithms. Rather than building a mediocre solution for both, I focused on the primary use case.'"

---

## Your Own Questions

### Q27: "What questions should I ask you?"

**Ask These to Show You're Thinking**:

1. **For Data Scientist Role**:
   > "How would you want me to monitor model drift if this goes into production?"

2. **For ML Engineer Role**:
   > "What infrastructure would be needed to scale this to 1,000 videos?"

3. **For AI Research Role**:
   > "Are there published techniques for handling extreme class imbalance better than information gain?"

4. **For PM/Product Role**:
   > "What would be the most valuable feature to add next - better accuracy on PPT, or supporting whiteboards?"

5. **For Full-Stack Role**:
   > "How would you build an API interface for this system?"

---

### Q28: "How do you stay updated with ML research?"

> "Three ways:

> 1. **Papers**: Read 2-3 ML papers per month from arxiv.org
> 2. **Projects**: Build small projects to test ideas (like this one)
> 3. **Community**: Follow researchers on Twitter, read blogs (Chris Olah, Lil'Log, etc.)

> But more importantly, I **learn by building**. This project taught me more about ML than reading 100 papers."

---

## Quick Reference Cheat Sheet

### Numbers to Remember
```
Test Accuracy:        97.45%
Precision:            77.25%
Recall:               79.63%
F1-Score:             78.42%
Real Validation:      93.6%
Class Imbalance:      40:1 (97.6% vs 2.4%)
Processing Speed:     10-15 min per 1-hour video
Total Frames:         41,650
Manual Transitions:   250
Videos:               14
```

### Key Phrases
- "The key insight was..."
- "The trade-off was..."
- "In retrospect, I learned..."
- "If I could do it again..."
- "The data clearly showed..."

### Red Flags to Avoid
❌ "I used the latest ML technique"
❌ "This is 97% accurate so it's perfect"
❌ "I tried 10 models and picked the best"
❌ "This problem is fully solved"

### Green Lights to Hit
✅ "I validated on real data"
✅ "Here's the trade-off analysis"
✅ "My constraint led to a learning"
✅ "I'm honest about limitations"

---

**Last Updated**: January 18, 2026  
**Status**: Use this file during interview prep! 🎯
