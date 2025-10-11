# OBS Studio Setup Guide for DiagXNet-Lite IDV

## 📥 **Download & Installation**
1. Download OBS Studio from: https://obsproject.com/
2. Install following standard installation process
3. Launch OBS Studio

---

## 🔧 **Initial Configuration**

### **Video Settings**
1. Go to **Settings** → **Video**
   - **Base (Canvas) Resolution:** Your screen resolution (likely 1920x1080 or 2560x1600)
   - **Output (Scaled) Resolution:** 1280x720 (good balance of quality/file size)
   - **Downscale Filter:** Lanczos (best quality)
   - **Common FPS Values:** 30 FPS

### **Audio Settings**  
1. Go to **Settings** → **Audio**
   - **Sample Rate:** 44.1 kHz
   - **Channels:** Stereo
   - **Desktop Audio:** Default (system sounds)
   - **Mic/Auxiliary Audio:** Your headset/microphone

### **Output Settings**
1. Go to **Settings** → **Output**
   - **Output Mode:** Simple
   - **Recording Quality:** High Quality, Medium File Size
   - **Recording Format:** MP4
   - **Encoder:** Hardware (if available) or Software (x264)

---

## 🎥 **Scene Setup**

### **Step 1: Add Display Capture**
1. In **Sources** box, click **+** 
2. Select **Display Capture**
3. Name it "Screen" and click **OK**
4. Select your display and click **OK**
5. This should fill the entire preview area

### **Step 2: Add Webcam**
1. Click **+** in Sources again
2. Select **Video Capture Device**  
3. Name it "Webcam" and click **OK**
4. Select your webcam device
5. Resolution: 640x480 or 1280x720
6. Click **OK**

### **Step 3: Position Webcam**
1. **Resize webcam:** Right-click webcam source → **Filters** → **+** → **Crop/Pad**
2. **Position webcam:** Drag to top-right corner
3. **Size:** Should be about 1/8 of screen (approximately 240x180 pixels)
4. **Test:** Make sure your face is visible and doesn't block important content

### **Step 4: Add Audio**
1. Click **+** in Sources
2. Select **Audio Input Capture**
3. Name it "Microphone" 
4. Select your headset/microphone
5. Test audio levels in the mixer section

---

## 🎚️ **Audio Level Setup**

### **Microphone Levels**
- **Target:** Green/yellow range (-20 dB to -12 dB)
- **Avoid:** Red (clipping/distortion)
- **Test:** Speak at normal recording volume while watching meter

### **Desktop Audio**
- **Lower system sounds** so they don't interfere with your voice
- **Mute notifications** during recording
- **Test:** Play a video to check audio balance

---

## 📐 **Layout Optimization**

### **Webcam Positioning Guidelines**
```
Screen Layout:
┌─────────────────────────────┬─────┐
│                             │ CAM │ ← 1/8 screen
│        Main Content         │     │
│                             ├─────┤
│                             │     │
│                             │     │
│                             │     │
└─────────────────────────────┴─────┘
```

### **Content Visibility Check**
- Open key files and ensure webcam doesn't block important text
- Check CSV files - numbers should be readable
- Test image viewing - charts should be fully visible
- Adjust webcam size/position if needed

---

## 🎬 **Recording Process**

### **Pre-Recording Steps**
1. **Close unnecessary apps** (notifications, background programs)
2. **Clear desktop** of sensitive/irrelevant files  
3. **Open key files** in order you'll use them
4. **Test audio** by recording 10-second sample
5. **Check webcam** - good lighting, clean background

### **During Recording**
1. Click **Start Recording** button
2. **Speak clearly** and at consistent pace
3. **Use mouse** to point at elements you're discussing
4. **Navigate smoothly** between applications
5. **Don't rush** - 10 minutes is sufficient time

### **After Recording**
1. Click **Stop Recording** 
2. **Review video** for technical issues
3. **Check audio sync** and quality
4. **Verify file size** (should be reasonable for upload)

---

## 🔍 **Quality Check Before Final Recording**

### **Video Quality Checklist**
- [ ] Screen content sharp and readable
- [ ] Webcam image clear with good lighting
- [ ] No important content hidden behind webcam
- [ ] Smooth mouse movements (not too fast)
- [ ] Application switching works smoothly

### **Audio Quality Checklist**  
- [ ] Voice clear and consistent volume
- [ ] No background noise or echo
- [ ] System sounds appropriately balanced
- [ ] No audio cutting out or distortion

### **Content Checklist**
- [ ] All demonstration files open correctly
- [ ] Python scripts run without errors
- [ ] CSV files display properly
- [ ] Images/charts load and display clearly
- [ ] Navigation between files is smooth

---

## 🚨 **Troubleshooting Common Issues**

### **Webcam Problems**
- **Not showing:** Check camera permissions in System Preferences
- **Poor quality:** Improve lighting, clean camera lens
- **Wrong size:** Right-click source → Transform → Edit Transform

### **Audio Problems**
- **No sound:** Check microphone permissions, select correct device
- **Too quiet/loud:** Adjust gain in OBS mixer
- **Echo:** Use headphones, move away from speakers

### **Performance Issues**
- **Choppy video:** Lower output resolution or frame rate
- **Large file size:** Use hardware encoder if available
- **Lag:** Close other applications, restart OBS

### **Screen Capture Issues**
- **Black screen:** Check display permissions, try different capture method
- **Wrong display:** Select correct monitor in display capture settings
- **Cursor missing:** Enable "Capture Cursor" in display capture

---

## 📊 **Recommended Settings Summary**

```
Video Settings:
- Canvas: Your screen resolution  
- Output: 1280x720
- FPS: 30
- Downscale: Lanczos

Audio Settings:
- Sample Rate: 44.1 kHz
- Bitrate: 160 kbps (AAC)

Layout:
- Screen: Full canvas
- Webcam: Top-right, ~240x180px
- Audio: Clear speech, minimal system sounds

File Output:
- Format: MP4
- Quality: High Quality, Medium File Size
- Target: Under 500MB for 10-minute video
```

---

**✅ You're ready to create an excellent IDV that showcases your DiagXNet-Lite achievements professionally and clearly!**