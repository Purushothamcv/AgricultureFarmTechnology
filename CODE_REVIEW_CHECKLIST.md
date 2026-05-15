Code Review Checklist: Memory Optimization
===========================================

This document helps you review the memory optimization changes for Render deployment.

## Files to Review

### 1. NEW FILE: `model_manager.py`
**What to check:**
- [ ] Imports: `joblib`, `tensorflow`, `threading`, `gc`
- [ ] Global cache `_model_cache` - empty dict ✓
- [ ] Thread lock for cache safety ✓
- [ ] Functions for lazy loading: `get_yield_model()`, `get_stress_model()`, etc.
- [ ] Memory cleanup: `cleanup_after_inference()`, `cleanup_all_models()`
- [ ] TensorFlow import deferred (inside function, not global)
- [ ] Error handling with None returns (doesn't crash if model missing)

**Key patterns:**
```python
def load_joblib_model(model_name, model_path):
    # Check cache first
    if model_name in _model_cache:
        return _model_cache[model_name]
    
    # Load and cache
    model = joblib.load(model_path)
    _model_cache[model_name] = model
    return model
```

### 2. NEW FILE: `logging_config.py`
**What to check:**
- [ ] Imports logging module
- [ ] Suppresses TensorFlow logger: `level=ERROR`
- [ ] Suppresses keras, matplotlib, PIL loggers
- [ ] Production mode detection
- [ ] Thread-safe logging configuration

**Should see:**
```python
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('keras').setLevel(logging.ERROR)
# ... more suppression
```

### 3. MODIFIED FILE: `main_fastapi.py`
**Lines 1-10 (TensorFlow Suppression):**
- [ ] `TF_CPP_MIN_LOG_LEVEL=3` set BEFORE any imports
- [ ] `TF_CPP_MIN_VLOG_LEVEL=3` set
- [ ] Comment explains why (before any imports)

**Around line 15 (Import logging_config):**
- [ ] `import logging_config` present
- [ ] Before model_manager import

**Around line 25-35 (Imports):**
- [ ] `from model_manager import ...` present
- [ ] Lists: `get_yield_model`, `get_stress_model`, etc.
- [ ] Lists: `cleanup_after_inference`, `cleanup_all_models`
- [ ] `suppress_tensorflow_logging()` called
- [ ] NO global `joblib.load()` calls
- [ ] NO direct TensorFlow imports
- [ ] NO model loading at module level

**Around line 200 (Startup Event):**
- [ ] LOW_MEMORY_MODE check present
- [ ] Skips heavy model loading if LOW_MEMORY_MODE=true
- [ ] Handles startup events in background

**Endpoints that use models:**
- [ ] `/predict_yield` - calls `get_yield_model()`
- [ ] `/predict_stress` - calls `get_stress_model()`
- [ ] `/recommend_crop` - calls `get_crop_model()`
- [ ] `/recommend_fertilizer` - calls `get_fert_model()`
- [ ] `/api/yield/predict` - uses lazy loading
- [ ] `/api/crop/recommend` - uses lazy loading
- [ ] Each calls `cleanup_after_inference()` after use

**New endpoint:**
- [ ] `/api/models/stats` - returns model cache status

**Shutdown event:**
- [ ] Calls `cleanup_all_models()` on shutdown

### 4. MODIFIED FILE: `requirements.txt`
**What to check:**
- [ ] No duplicate packages
- [ ] All versions pinned (e.g., `==2.5.0` not `>=2.5.0`)
- [ ] Removed packages: streamlit, matplotlib, seaborn, folium
- [ ] Kept: fastapi, uvicorn, pymongo, motor, tensorflow, keras
- [ ] No unused packages added

**Should NOT see:**
```
streamlit
matplotlib
seaborn
folium
streamlit-folium
```

**Should see:**
```
fastapi==0.104.1
uvicorn==0.24.0
tensorflow==2.14.0
keras==2.14.0
pymongo==4.6.0
motor==3.3.2
```

### 5. MODIFIED FILE: `Dockerfile`
**What to check:**
- [ ] Build stage: `FROM python:3.10-slim`
- [ ] Environment variables set:
  - [ ] `PYTHONUNBUFFERED=1`
  - [ ] `PYTHONDONTWRITEBYTECODE=1`
  - [ ] `LOW_MEMORY_MODE=true`
  - [ ] `ENVIRONMENT=production`
  - [ ] `TF_CPP_MIN_LOG_LEVEL=3`
- [ ] CMD starts uvicorn with:
  - [ ] `--workers 1` (single worker)
  - [ ] `--timeout-keep-alive 5` (aggressive cleanup)
  - [ ] `--timeout-notify 30` (faster shutdown)

**Should see:**
```dockerfile
ENV LOW_MEMORY_MODE=true
ENV ENVIRONMENT=production
CMD ["sh", "-c", "python -m uvicorn main_fastapi:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 5 --timeout-notify 30"]
```

### 6. MODIFIED FILE: `render.yaml`
**What to check:**
- [ ] Environment variables added:
  - [ ] `LOW_MEMORY_MODE=true`
  - [ ] `ENVIRONMENT=production`
  - [ ] `TF_CPP_MIN_LOG_LEVEL=3`
- [ ] MongoDB URL variable present
- [ ] Google OAuth secrets present
- [ ] GROQ_API_KEY present (optional)

## Testing the Changes

### Test 1: No Global Model Loads
```bash
# Should NOT see these patterns:
grep -n "joblib.load" backend/main_fastapi.py  # Should be 0 matches
grep -n "^yield_model = " backend/main_fastapi.py  # Should be 0 matches
grep -n "import tensorflow" backend/main_fastapi.py  # Should be 0 matches
```

### Test 2: Model Manager Imports
```bash
# Should see this pattern:
grep -n "from model_manager import" backend/main_fastapi.py  # Should have matches
grep -n "get_yield_model" backend/main_fastapi.py  # Should have matches
```

### Test 3: Lazy Loading in Endpoints
```bash
# Each endpoint should call get_*_model()
grep -A5 "def predict_yield" backend/main_fastapi.py | grep "get_yield_model"
grep -A5 "def predict_stress" backend/main_fastapi.py | grep "get_stress_model"
```

### Test 4: Memory Cleanup
```bash
# Should see cleanup calls
grep -n "cleanup_after_inference" backend/main_fastapi.py  # Should have matches
grep -n "cleanup_all_models" backend/main_fastapi.py  # Should have matches
```

### Test 5: TensorFlow Suppression
```bash
# Should be very first lines
head -10 backend/main_fastapi.py | grep "TF_CPP_MIN_LOG_LEVEL"
```

## Runtime Verification

### Test: Startup Time
```bash
# Should complete in 3-5 seconds
time python -m uvicorn backend/main_fastapi:app --port 8000

# Look for: "[OK] Port ready - services loading in background"
```

### Test: Memory at Startup
```bash
# Should be 50-100MB initially
ps aux | grep uvicorn | grep -v grep

# After loading models, should stay under 400MB
```

### Test: Lazy Loading Works
```bash
# Check initial state
curl http://localhost:8000/api/models/stats
# Result: "cached_models": []

# Make prediction (loads model)
curl "http://localhost:8000/recommend_crop?..."

# Check after
curl http://localhost:8000/api/models/stats
# Result: "cached_models": ["crop_model"]
```

## Code Quality Checklist

- [ ] No syntax errors
- [ ] All imports are valid
- [ ] No circular imports
- [ ] Type hints used where appropriate
- [ ] Docstrings on new functions
- [ ] Comments explain non-obvious code
- [ ] No debug print statements left
- [ ] Error handling present
- [ ] Graceful degradation (no crashes)
- [ ] Thread-safe operations
- [ ] No hardcoded paths (uses relative paths)

## Security Checklist

- [ ] No secrets in code
- [ ] All secrets use environment variables
- [ ] No passwords in files
- [ ] API keys loaded from .env
- [ ] No SQL injection vectors
- [ ] No command injection in subprocess calls
- [ ] Input validation present
- [ ] Error messages don't leak info

## Performance Checklist

- [ ] Startup time reduced
- [ ] Memory reduced at startup
- [ ] No unnecessary imports
- [ ] Models cached efficiently
- [ ] Garbage collection called appropriately
- [ ] No memory leaks detected
- [ ] Endpoints respond quickly
- [ ] Database queries optimized

## Compatibility Checklist

- [ ] No breaking API changes
- [ ] Endpoints return same format
- [ ] Request parameters unchanged
- [ ] Database schema unchanged
- [ ] Authentication unchanged
- [ ] Frontend integration unchanged
- [ ] Can rollback safely
- [ ] Migrations not needed

## Documentation Checklist

- [ ] README updated (if needed)
- [ ] Comments in code explain why
- [ ] Deployment guide provided
- [ ] Troubleshooting guide provided
- [ ] Configuration documented
- [ ] Environment variables documented
- [ ] Performance metrics documented
- [ ] Migration path documented

## Sign-Off

When all checks pass:

- [ ] Code review complete
- [ ] Tests pass locally
- [ ] No memory issues observed
- [ ] Performance improved
- [ ] Documentation sufficient
- [ ] Ready for production deployment
- [ ] Approved for merge

---

## Review Summary

**Changed Files**: 8  
**New Files**: 4  
**Documentation**: 6 guides  
**Breaking Changes**: None ✓  
**Backward Compatible**: Yes ✓  

**Status**: ✓ Ready for Review  
**Risk Level**: Low  
**Rollback**: Safe (git revert)

---

Reviewed by: ________________  
Date: ________________  
Approved: □ Yes  □ No  □ Conditional  

Comments:
