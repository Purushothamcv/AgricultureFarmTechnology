# 🌿 Plant Disease Detection - Quick Reference

## ⚡ Quick Summary

**Added:** Crop selection dropdown + confidence validation (55% threshold)
**Impact:** Better accuracy, improved user experience

---

## 🎯 What Users See

### Crop Dropdown
```
┌─ Select Crop Type ───────────┐
│ -- Select a crop --          │
│ Apple                        │
│ Blueberry                    │
│ Cherry                       │
│ Corn (Maize)                 │
│ ... and 10 more              │
└──────────────────────────────┘
```

### Valid Prediction (≥ 55%)
```
✓ HEALTHY / ⚠ DISEASE DETECTED

Crop: Tomato
Status: Early Blight
Confidence: 87.3% [████████░]
```

### Low Confidence (< 55%)
```
⚠️ LOW CONFIDENCE DETECTION

"Please upload a clearer and valid plant leaf image."
Confidence: 42.3%
```

---

## 📊 Validation Decision Tree

```
Crop Selected?
├─ NO → Error: "Please select a crop type"
└─ YES ↓
    Crop Supported?
    ├─ NO → Error: "Not supported"
    └─ YES ↓
        Image Uploaded?
        ├─ NO → Error: "Upload image"
        └─ YES ↓
            Valid Crop Label?
            ├─ NO → Error: "Invalid image"
            └─ YES ↓
                Crop Matches?
                ├─ NO → Error: "Doesn't match"
                └─ YES ↓
                    Confidence ≥ 55%?
                    ├─ NO → Show Warning
                    └─ YES ↓
                        Display Results ✓
```

---

## 🧪 Quick Test Checklist

- [ ] Select crop, upload matching image
- [ ] Don't select crop, try to upload
- [ ] Select crop, upload different crop image
- [ ] Upload low-confidence/blurry image
- [ ] Upload high-confidence clear image
- [ ] Upload non-plant image
- [ ] Test with healthy leaf image
- [ ] Test with diseased leaf image

---

## 💾 Supported Crops (14)

Apple • Blueberry • Cherry • Corn (Maize) • Grape • Orange • Peach • Pepper (Bell) • Potato • Raspberry • Soybean • Squash • Strawberry • Tomato

---

## ⚠️ Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "Please select a crop type" | No selection | Pick a crop |
| "Not supported by model" | Wrong crop | Select supported crop |
| "Invalid image" | Non-plant image | Upload plant leaf |
| "Doesn't match selected crop" | Wrong crop image | Match crop type |
| "Low confidence detected" | Unclear image | Upload clearer image |

---

## ✅ Success Messages

| Status | Display |
|--------|---------|
| Healthy plant | ✓ Healthy (No Disease) |
| Disease detected | ⚠ Disease name shown |
| High confidence | Full details + progress bar |

---

## 🎨 Visual Indicators

**Healthy Leaf:**
- 🟢 Green background
- ✓ Healthy (No Disease Detected)
- Confidence bar (usually high)

**Disease Detected:**
- 🟠 Orange background
- ⚠ Disease name
- Confidence bar

**Low Confidence:**
- 🟡 Yellow alert box
- ⚠️ Warning icon
- Request for clearer image

---

## 🔧 Configuration

**Confidence Threshold:** 55%
**Location:** `frontend/src/pages/LeafDisease.jsx`
**Line:** ~134

To adjust:
```javascript
const CONFIDENCE_THRESHOLD = 0.55; // Change this
```

---

## 📋 Label Format

**Input Format:** 
- `Crop___Disease` (e.g., `Tomato___Early_blight`)
- `Disease_Crop` (e.g., `Powdery_mildew_Grape`)

**Parsing:**
```
Tomato___Late_blight → Crop: Tomato, Disease: Late_blight
Early_blight_Tomato → Crop: Tomato, Disease: Early_blight
```

---

## 🚀 Workflow

1. **Open Plant Disease Page**
2. **Select Crop** (mandatory)
3. **Upload Leaf Image**
4. **Click Detect Disease**
5. **Review Results**

---

## ✨ Key Improvements

✅ Mandatory crop selection
✅ Confidence threshold blocking low-accuracy predictions
✅ Crop matching validation
✅ Yellow warning for low confidence
✅ Clear error messages
✅ Better result display format

---

## 📞 FAQ

**Q: Why do I need to select a crop?**
A: It helps validate predictions and ensures image matches the crop type.

**Q: What's the confidence threshold?**
A: 55%. Below that, the model isn't confident enough, so we show a warning.

**Q: Can I change the threshold?**
A: Yes, but 55% is optimal for agriculture decision-making.

**Q: What if my crop isn't listed?**
A: Only 14 crops are currently trained. Others coming soon!

---

**Status**: ✅ Live
**Version**: 2.0.0
**Crops**: 14 supported
**Threshold**: 55% confidence
