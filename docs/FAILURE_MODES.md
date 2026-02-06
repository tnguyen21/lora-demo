# Common Failure Modes & Debugging Guide

## 1. Bad Teacher Labels

**Symptoms:** Student model learns wrong patterns, accuracy plateaus below teacher accuracy.

**Causes:**
- Teacher prompt is ambiguous or has contradictory instructions
- Teacher model isn't capable enough for the task
- Edge cases not covered in the prompt

**Fixes:**
- Spot-check 50-100 teacher labels before training
- Compute teacher accuracy against any ground truth you have
- If teacher accuracy < 90%, fix the prompt first — don't train on bad labels
- Add explicit edge case rules to the prompt

## 2. Learning Rate Too High

**Symptoms:** Training loss spikes or oscillates. Eval accuracy drops after initial improvement.

**Fixes:**
- Start with `1e-4` for LoRA fine-tuning (this is ~10x higher than full fine-tuning, which is correct for LoRA)
- If loss spikes, try `5e-5` or `3e-5`
- Use `lr_schedule="linear"` or `"cosine"` for decay
- Check the training logs for loss curves

## 3. Learning Rate Too Low

**Symptoms:** Training loss decreases very slowly. Model barely changes from base behavior.

**Fixes:**
- For LoRA, `1e-4` to `5e-4` is the typical range
- If loss barely moves after 1 epoch, increase by 2-5x
- Make sure you're using LoRA-appropriate LR (not full fine-tuning LR)

## 4. Tokenizer / Renderer Mismatch

**Symptoms:** Garbled output, special tokens in predictions, model outputs nonsense.

**Causes:**
- Using wrong renderer for the model family
- Tokenizer doesn't match the model

**Fixes:**
- Qwen3 text models → `renderer_name="qwen3"`
- Qwen3 VL models → `renderer_name="qwen3_vl"`
- Use `model_info.get_recommended_renderer_name(model_name)` to auto-detect
- Verify tokenizer with `get_tokenizer(model_name)`

## 5. Output Format Drift

**Symptoms:** Model outputs are almost correct but in wrong format (extra text, missing quotes, wrong JSON structure).

**Causes:**
- Teacher output isn't consistent across examples
- Training data has mixed formats

**Fixes:**
- Ensure teacher prompt specifies exact output format
- Post-process teacher output before saving to JSONL
- Use `output_regex` to validate teacher responses
- Filter out malformed examples before training

## 6. Overfitting (Too Many Epochs)

**Symptoms:** Training loss near 0, but eval accuracy decreases. Model memorizes training examples instead of generalizing.

**Fixes:**
- For classification tasks, 2-4 epochs is usually sufficient
- Monitor eval metrics during training (use `eval_every` parameter)
- If eval accuracy peaks and drops, use the checkpoint from that peak
- Increase training data if possible

## 7. Not Enough Training Data

**Symptoms:** High variance in predictions, poor generalization to new inputs.

**Rules of thumb:**
- Classification (2-10 labels): 200-1000 examples
- Extraction (structured JSON): 500-2000 examples
- Generation (SQL, free text): 1000-5000 examples
- Vision classification: 50-200 per class

**Fixes:**
- Generate more synthetic data with the teacher model
- Augment existing data (paraphrase, add noise)
- Use `synthetic_examples` in the config to control data volume

## 8. Checkpoint Not Loading

**Symptoms:** Model behaves like base model, ignoring fine-tuning.

**Causes:**
- Wrong checkpoint path
- Checkpoint expired (default TTL is 7 days)
- Using `base_model` instead of `model_path` when loading

**Fixes:**
- Use `model_path="tinker://<run-id>/sampler_weights/final"` (not `base_model`)
- Check checkpoint exists: look in the training log directory for `checkpoints.jsonl`
- Re-train if checkpoint expired

## 9. Vision Model Issues

**Symptoms:** VLM ignores image content, classifies based on text only.

**Causes:**
- Image not properly encoded/resized
- Wrong image format
- Image too large (exceeds max token budget)

**Fixes:**
- Use `resize_image(image, max_size=480)` to control image token count
- Ensure images are RGB (convert RGBA/grayscale)
- Verify images load correctly with PIL before sending to model
- Check that `ImagePart` is in the message content
