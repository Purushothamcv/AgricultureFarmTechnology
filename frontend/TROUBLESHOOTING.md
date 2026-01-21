# 🔧 Smart Agri Frontend - Troubleshooting Guide

## Common Issues & Solutions

---

## 🚫 Installation Issues

### Issue 1: "npm: command not found" or "node: command not found"

**Problem:** Node.js is not installed or not in PATH

**Solutions:**
```powershell
# Option 1: Install Node.js
# Download from: https://nodejs.org/
# Choose LTS version (18.x or higher)

# Option 2: Verify installation
node --version    # Should show v18.x.x or higher
npm --version     # Should show 9.x.x or higher

# Option 3: Add to PATH (Windows)
# Search "Environment Variables" → Edit PATH
# Add: C:\Program Files\nodejs\
```

---

### Issue 2: "npm install" fails with errors

**Problem:** Network issues, permission errors, or corrupted cache

**Solutions:**
```powershell
# Clear npm cache
npm cache clean --force

# Delete existing node_modules and lock file
Remove-Item -Recurse -Force node_modules
Remove-Item package-lock.json

# Reinstall
npm install

# If still fails, try with --legacy-peer-deps
npm install --legacy-peer-deps

# Or use different registry
npm install --registry=https://registry.npmjs.org/
```

---

### Issue 3: Permission denied during install

**Problem:** Insufficient permissions

**Solutions:**
```powershell
# Run PowerShell as Administrator
# Right-click PowerShell → "Run as Administrator"

# Or change npm global folder
npm config set prefix %APPDATA%\npm

# On Linux/Mac, avoid sudo
npm config set prefix ~/.npm-global
```

---

## 🖥️ Development Server Issues

### Issue 4: Port 3000 already in use

**Problem:** Another application is using port 3000

**Solutions:**
```powershell
# Option 1: Kill process on port 3000
netstat -ano | findstr :3000
# Note the PID, then:
taskkill /PID <PID> /F

# Option 2: Change port in vite.config.js
# Edit vite.config.js:
server: {
  port: 3001,  # Use different port
}

# Option 3: Start with different port
npm run dev -- --port 3001
```

---

### Issue 5: "npm run dev" starts but browser shows blank page

**Problem:** Build error, JavaScript disabled, or cache issues

**Solutions:**
```powershell
# Check browser console for errors (F12)

# Clear browser cache:
# Chrome: Ctrl+Shift+Delete
# Edge: Ctrl+Shift+Delete
# Firefox: Ctrl+Shift+Delete

# Hard reload:
Ctrl+Shift+R (Chrome/Edge)
Ctrl+F5 (Firefox)

# Delete .vite cache
Remove-Item -Recurse -Force node_modules\.vite

# Restart dev server
npm run dev
```

---

### Issue 6: Hot reload not working

**Problem:** File watcher not detecting changes

**Solutions:**
```powershell
# Increase file watchers (Windows)
# Add to vite.config.js:
server: {
  watch: {
    usePolling: true
  }
}

# Save file again
# Or restart dev server
```

---

## 🗺️ Map & Leaflet Issues

### Issue 7: Map not displaying, shows gray box

**Problem:** Leaflet CSS not loaded or tiles not loading

**Solutions:**
```html
<!-- Verify in index.html: -->
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />

<!-- Check browser console for CSS errors -->

<!-- Ensure internet connection (map uses online tiles) -->
```

**Alternative fix:**
```powershell
# Clear browser cache
# Check network tab in DevTools (F12)
# Look for failed tile requests
```

---

### Issue 8: Map markers not showing

**Problem:** Marker icon paths incorrect

**Solution in Dashboard.jsx:**
```jsx
// Already fixed in code, but if issues persist:
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});
```

---

## 🔐 Authentication Issues

### Issue 9: Login fails with "Network Error"

**Problem:** Backend not running or incorrect API URL

**Solutions:**
```powershell
# Check .env file exists
ls .env

# Verify API URL
Get-Content .env
# Should show: VITE_API_BASE_URL=http://localhost:8000/api

# Check if backend is running
# Open http://localhost:8000/api in browser

# Update .env if needed
"VITE_API_BASE_URL=http://localhost:8000/api" | Set-Content .env

# Restart dev server (important!)
# Stop server (Ctrl+C)
npm run dev
```

---

### Issue 10: "401 Unauthorized" after login

**Problem:** Token expired or invalid

**Solutions:**
```powershell
# Clear localStorage in browser:
# F12 → Application → Local Storage → Clear All

# Or in browser console:
localStorage.clear()

# Login again
```

---

### Issue 11: Stuck on login page after successful login

**Problem:** Redirect not working

**Solutions:**
```jsx
// Check browser console for errors

// Verify in AuthContext.jsx:
const login = async (credentials) => {
  const data = await authService.login(credentials);
  setUser(data.user);  // Make sure this is called
  return { success: true };
};

// Manually navigate in browser:
# Go to http://localhost:3000/dashboard
```

---

## 🌐 API & Network Issues

### Issue 12: CORS errors in console

**Problem:** Backend doesn't allow frontend origin

**Backend Solution:**
```python
# In your FastAPI/Flask backend, add:
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

### Issue 13: API calls return 404

**Problem:** Wrong endpoint or API not running

**Solutions:**
```powershell
# Check API is running
curl http://localhost:8000/api/health

# Verify endpoint in services.js
# Example:
await api.post('/crop/recommend', data);
# Full URL: http://localhost:8000/api/crop/recommend

# Check backend logs for errors
```

---

### Issue 14: Image upload fails

**Problem:** File size, type, or backend configuration

**Solutions:**
```jsx
// Check file size (< 10MB)
// Check file type (image/png, image/jpeg, image/jpg)

// In backend, ensure multipart/form-data is accepted:
@app.post("/api/disease/fruit")
async def detect_disease(image: UploadFile = File(...)):
    # ...
```

---

## 🎨 Styling Issues

### Issue 15: Tailwind classes not applying

**Problem:** Tailwind not configured or build issue

**Solutions:**
```powershell
# Verify tailwind.config.js exists and has:
content: [
  "./index.html",
  "./src/**/*.{js,ts,jsx,tsx}",
]

# Check postcss.config.js exists

# Restart dev server
npm run dev

# If still not working, rebuild:
Remove-Item -Recurse -Force node_modules\.vite
npm run dev
```

---

### Issue 16: Custom CSS not loading

**Problem:** Import order or CSS file missing

**Solutions:**
```jsx
// Verify in main.jsx:
import './index.css'

// Check index.css has:
@tailwind base;
@tailwind components;
@tailwind utilities;

// Clear browser cache
```

---

## 📱 Responsive Design Issues

### Issue 17: Mobile menu not working

**Problem:** JavaScript state issue

**Solutions:**
```jsx
// Check in Navbar.jsx:
const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

// Verify button onClick:
onClick={() => setMobileMenuOpen(!mobileMenuOpen)}

// Check browser console for errors
```

---

### Issue 18: Layout broken on mobile

**Problem:** Fixed widths or overflow

**Solutions:**
```jsx
// Use Tailwind responsive classes:
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3">

// Test in browser DevTools:
// F12 → Toggle device toolbar (Ctrl+Shift+M)
// Test different screen sizes
```

---

## 🖼️ Image & Asset Issues

### Issue 19: Images not loading

**Problem:** Wrong path or missing files

**Solutions:**
```jsx
// Place images in public/ folder
// Reference as: /image.png (not ./image.png)

// Or import in component:
import logo from './assets/logo.png'
<img src={logo} alt="Logo" />
```

---

## ⚡ Performance Issues

### Issue 20: Slow page load

**Problem:** Large bundle size or slow API

**Solutions:**
```powershell
# Build and analyze
npm run build

# Check dist/ folder size
# Should be < 1MB for JS

# Lazy load routes:
const Dashboard = lazy(() => import('./pages/Dashboard'));

# Optimize images before uploading
```

---

### Issue 21: High memory usage

**Problem:** Memory leak or too many re-renders

**Solutions:**
```jsx
// Use React DevTools Profiler

// Add cleanup in useEffect:
useEffect(() => {
  const subscription = subscribe();
  return () => subscription.unsubscribe();
}, []);

// Restart dev server
```

---

## 🏗️ Build Issues

### Issue 22: "npm run build" fails

**Problem:** TypeScript errors or import issues

**Solutions:**
```powershell
# Check console errors

# Clear cache
Remove-Item -Recurse -Force node_modules\.vite
Remove-Item -Recurse -Force dist

# Try building again
npm run build

# If ESLint errors, temporarily disable:
# In vite.config.js, comment out ESLint plugin

# Fix errors one by one
```

---

### Issue 23: Build succeeds but preview fails

**Problem:** Route configuration or base URL

**Solutions:**
```powershell
# Preview build
npm run preview

# Check vite.config.js:
# Make sure no custom base path unless needed

# Verify all routes work in preview
```

---

## 🔍 Debugging Tips

### General Debugging Workflow

```powershell
1. Check Browser Console (F12)
   - Look for red errors
   - Check Network tab for failed requests

2. Check Terminal Output
   - Look for build errors
   - Check for warnings

3. Clear Everything
   Remove-Item -Recurse -Force node_modules
   Remove-Item -Recurse -Force node_modules\.vite
   Remove-Item -Recurse -Force dist
   Remove-Item package-lock.json
   npm install
   npm run dev

4. Test in Incognito Mode
   - Eliminates cache/extension issues

5. Check Git Status
   git status
   # Make sure you have all files

6. Verify Environment
   Get-Content .env
   # Should have: VITE_API_BASE_URL=...
```

---

## 🆘 Emergency Reset

### Nuclear Option: Complete Reset

```powershell
# Stop all running servers
# Ctrl+C in all terminals

# Delete everything
Remove-Item -Recurse -Force node_modules
Remove-Item -Recurse -Force dist
Remove-Item -Recurse -Force node_modules\.vite
Remove-Item package-lock.json

# Fresh install
npm install

# Clear browser completely
# Chrome: chrome://settings/clearBrowserData
# Clear "Cached images and files"
# Clear "Cookies and other site data"

# Restart dev server
npm run dev

# Test in fresh incognito window
```

---

## 📞 Getting Help

### Before Asking for Help

Collect this information:

1. **Error Message:**
   - Full error from console
   - Screenshot if visual issue

2. **Environment:**
   ```powershell
   node --version
   npm --version
   Get-Content package.json | Select-String version
   ```

3. **Steps to Reproduce:**
   - What did you do?
   - What happened?
   - What did you expect?

4. **What You've Tried:**
   - List solutions you attempted

### Where to Get Help

- **GitHub Issues**: Post detailed bug reports
- **Stack Overflow**: Tag with `reactjs`, `vite`, `tailwindcss`
- **Discord Communities**: React, Tailwind CSS servers
- **Documentation**: Check README.md, SETUP_GUIDE.md

---

## ✅ Health Check Checklist

Run this checklist to verify everything is working:

```
✅ Node.js installed (node --version)
✅ npm working (npm --version)
✅ Dependencies installed (ls node_modules)
✅ .env file exists (ls .env)
✅ Dev server starts (npm run dev)
✅ Browser opens to localhost:3000
✅ Login page loads
✅ Can navigate to dashboard
✅ Map displays
✅ No console errors
✅ API calls work (or demo data shows)
✅ All modules accessible
✅ Responsive on mobile (F12 → device toggle)
✅ Build succeeds (npm run build)
```

---

## 🎯 Prevention Tips

### Best Practices to Avoid Issues

1. **Always use exact versions** in package.json
2. **Commit package-lock.json** to git
3. **Don't modify node_modules** directly
4. **Keep Node.js updated** (LTS version)
5. **Clear cache regularly** when issues occur
6. **Test in multiple browsers**
7. **Use .env for configuration**, never hardcode
8. **Restart dev server** after config changes
9. **Check console** before reporting bugs
10. **Read error messages** completely

---

## 📚 Useful Commands Reference

```powershell
# Check versions
node --version
npm --version

# Clean installs
npm ci                          # Clean install from lock file
npm install --force             # Force reinstall

# Clear caches
npm cache clean --force         # Clear npm cache
Remove-Item -Recurse -Force node_modules\.vite   # Clear Vite cache

# Development
npm run dev                     # Start dev server
npm run dev -- --port 3001      # Start on different port
npm run dev -- --host          # Expose to network

# Production
npm run build                   # Build for production
npm run preview                 # Preview production build

# Debugging
npm run dev -- --debug          # Debug mode
npm ls                          # List installed packages
npm outdated                    # Check for updates

# Package management
npm install <package>           # Install new package
npm uninstall <package>         # Remove package
npm update                      # Update packages
```

---

**Still having issues? Check the documentation:**
- [README.md](./README.md) - Overview
- [SETUP_GUIDE.md](./SETUP_GUIDE.md) - Detailed setup
- [PROJECT_SUMMARY.md](./PROJECT_SUMMARY.md) - Complete guide
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Architecture details

---

**Remember: Most issues are solved by:**
1. Reading the error message
2. Clearing cache
3. Restarting the dev server
4. Checking the browser console

Good luck! 🚀
