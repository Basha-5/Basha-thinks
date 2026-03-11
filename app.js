/* ═══════════════════════════════════════════════════════════════
   BioAuth — Behavioral Biometrics System
   app.js
   ═══════════════════════════════════════════════════════════════ */

// ─── Constants ────────────────────────────────────────────────────────────────
const DB_KEY     = 'bioauth_users_v2';
const MODELS_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model/';
const THRESHOLD  = 0.55;   // Euclidean distance cutoff for face match
const SAMPLES    = 5;      // Number of face descriptor samples to average

// ─── State ────────────────────────────────────────────────────────────────────
let db                   = JSON.parse(localStorage.getItem(DB_KEY) || '{}');
let regStream            = null;
let loginStream          = null;
let regDescriptors       = [];
let regDetectionInterval = null;
let loginDetectionInterval = null;
let modelsLoaded         = false;

// ═══════════════════════════════════════════════════════════════
// PARTICLES
// ═══════════════════════════════════════════════════════════════

(function createParticles() {
  const container = document.getElementById('particles');

  for (let i = 0; i < 30; i++) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    particle.style.left            = Math.random() * 100 + '%';
    particle.style.animationDuration  = (8 + Math.random() * 12) + 's';
    particle.style.animationDelay    = (Math.random() * 15) + 's';
    particle.style.opacity           = Math.random() * 0.5;

    // Some particles are green and slightly bigger
    if (Math.random() > 0.7) {
      particle.style.background = 'var(--green)';
      particle.style.width  = '3px';
      particle.style.height = '3px';
    }

    container.appendChild(particle);
  }
})();

// ═══════════════════════════════════════════════════════════════
// MODEL LOADING
// ═══════════════════════════════════════════════════════════════

async function loadModels() {
  const msgEl = document.getElementById('loadMsg');

  try {
    msgEl.textContent = 'Loading TinyFaceDetector...';
    await faceapi.nets.tinyFaceDetector.loadFromUri(MODELS_URL);

    msgEl.textContent = 'Loading FaceLandmark68Net...';
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODELS_URL);

    msgEl.textContent = 'Loading FaceRecognitionNet...';
    await faceapi.nets.faceRecognitionNet.loadFromUri(MODELS_URL);

    modelsLoaded = true;
    document.getElementById('loadingOverlay').style.display = 'none';
    document.getElementById('systemStatus').textContent = 'BIOMETRICS READY';

  } catch (err) {
    msgEl.textContent = 'ERROR: ' + err.message;
    msgEl.style.color = 'var(--red)';
    console.error('Model load error:', err);

    // Still dismiss overlay after a short delay so the UI remains usable
    setTimeout(() => {
      document.getElementById('loadingOverlay').style.display = 'none';
    }, 2500);
  }
}

// Kick off model loading immediately
loadModels();

// ═══════════════════════════════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════════════════════════════

/**
 * Switch the visible page and update nav highlight.
 * Also stops cameras when leaving their respective pages.
 * @param {string} id - 'home' | 'register' | 'login' | 'about'
 */
function showPage(id) {
  const pageOrder = ['home', 'register', 'login', 'about'];

  // Deactivate all pages and buttons
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));

  // Activate target page and matching nav button
  document.getElementById('page-' + id).classList.add('active');
  document.querySelectorAll('.nav-btn')[pageOrder.indexOf(id)].classList.add('active');

  // Refresh enrolled-users list when navigating to login
  if (id === 'login') updateUsersList();

  // Stop cameras when navigating away from their pages
  if (id !== 'register') stopRegCamera();
  if (id !== 'login')    stopLoginCamera();
}

// ═══════════════════════════════════════════════════════════════
// ALERTS
// ═══════════════════════════════════════════════════════════════

/**
 * Render a dismissing alert inside a container element.
 * @param {string} containerId - Element id (e.g. 'reg-alerts')
 * @param {'success'|'error'|'info'|'warning'} type
 * @param {string} message
 */
function showAlert(containerId, type, message) {
  const icons = { success: '✓', error: '✕', info: '◈', warning: '⚠' };
  const el    = document.getElementById(containerId);

  el.innerHTML = `
    <div class="alert ${type}">
      ${icons[type] || '◈'} ${message}
    </div>`;

  // Auto-dismiss after 5 seconds
  setTimeout(() => { el.innerHTML = ''; }, 5000);
}

// ═══════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════

/**
 * Hash a plaintext password with SHA-256.
 * Returns a hex string.
 */
async function hashPassword(pw) {
  const buffer = await crypto.subtle.digest(
    'SHA-256',
    new TextEncoder().encode(pw)
  );
  return Array.from(new Uint8Array(buffer))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

/** Persist the in-memory user database to localStorage. */
function saveDB() {
  localStorage.setItem(DB_KEY, JSON.stringify(db));
}

/**
 * Average an array of Float32Array face descriptors into a single descriptor.
 * @param {Float32Array[]} descriptors
 * @returns {Float32Array}
 */
function averageDescriptors(descriptors) {
  const length = descriptors[0].length;
  const avg    = new Float32Array(length);

  descriptors.forEach(desc => {
    for (let i = 0; i < length; i++) avg[i] += desc[i];
  });

  for (let i = 0; i < length; i++) avg[i] /= descriptors.length;

  return avg;
}

// ─── Status Helpers ───────────────────────────────────────────────────────────

/** Update the registration page face-status indicator. */
function setRegStatus(type, text) {
  document.getElementById('regDot').className = 'face-status-dot ' + type;
  document.getElementById('regStatusText').textContent = text;
}

/** Update the login page face-status indicator. */
function setLoginStatus(type, text) {
  document.getElementById('loginDot').className = 'face-status-dot ' + type;
  document.getElementById('loginStatusText').textContent = text;
}

// ═══════════════════════════════════════════════════════════════
// REGISTRATION — CAMERA
// ═══════════════════════════════════════════════════════════════

/** Request webcam access and start the registration preview. */
async function startRegCamera() {
  if (!modelsLoaded) {
    showAlert('reg-alerts', 'warning', 'Biometric models still loading, please wait...');
    return;
  }

  try {
    regStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' }
    });

    const video = document.getElementById('regVideo');
    video.srcObject = regStream;
    await video.play();

    document.getElementById('regStartCam').textContent = '■ CAMERA ON';
    document.getElementById('regCapture').disabled = false;
    setRegStatus('scanning', 'Scanning for face...');
    startRegDetection();

  } catch (err) {
    showAlert('reg-alerts', 'error', 'Camera access denied: ' + err.message);
  }
}

/** Stop the registration camera and clear the canvas. */
function stopRegCamera() {
  if (regStream) {
    regStream.getTracks().forEach(t => t.stop());
    regStream = null;
  }

  if (regDetectionInterval) {
    clearInterval(regDetectionInterval);
    regDetectionInterval = null;
  }

  document.getElementById('regStartCam').textContent = '📷 START CAMERA';
  document.getElementById('regCapture').disabled = true;
  setRegStatus('idle', 'Camera not started');

  // Clear canvas overlay
  const canvas = document.getElementById('regCanvas');
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

/**
 * Run continuous face detection on the registration video feed
 * and draw bounding box + landmark dots onto the canvas overlay.
 */
function startRegDetection() {
  const video  = document.getElementById('regVideo');
  const canvas = document.getElementById('regCanvas');

  regDetectionInterval = setInterval(async () => {
    if (!video.videoWidth) return;

    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;

    const detection = await faceapi
      .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptor()
      .catch(() => null);

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (detection) {
      const { x, y, width, height } = detection.detection.box;

      // Bounding box
      ctx.strokeStyle = '#00ff88';
      ctx.lineWidth   = 2;
      ctx.strokeRect(x, y, width, height);

      // Landmark dots
      ctx.fillStyle = 'rgba(0, 255, 136, 0.7)';
      detection.landmarks.positions.forEach(pt => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 1.5, 0, Math.PI * 2);
        ctx.fill();
      });

      // Label
      ctx.fillStyle = '#00ff88';
      ctx.font      = '11px Share Tech Mono';
      ctx.fillText('FACE DETECTED', x, y - 8);

      setRegStatus('detected', 'Face detected — ready to capture');

    } else {
      setRegStatus('scanning', 'No face detected — position your face in frame');
    }

  }, 200); // ~5fps for the overlay
}

// ═══════════════════════════════════════════════════════════════
// REGISTRATION — FACE CAPTURE
// ═══════════════════════════════════════════════════════════════

/**
 * Capture SAMPLES frames from the webcam, extract face descriptors,
 * and average them into a single enrollment template.
 */
async function captureRegFace() {
  const video = document.getElementById('regVideo');

  if (!video.videoWidth) {
    showAlert('reg-alerts', 'error', 'Camera not active');
    return;
  }

  // Disable button during capture
  document.getElementById('regCapture').disabled = true;
  document.getElementById('regProgress').style.display = 'block';
  setRegStatus('scanning', 'Capturing samples...');

  regDescriptors = [];

  for (let i = 0; i < SAMPLES; i++) {
    const detection = await faceapi
      .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 416 }))
      .withFaceLandmarks()
      .withFaceDescriptor()
      .catch(() => null);

    if (!detection) {
      showAlert('reg-alerts', 'error', `Sample ${i + 1} failed — ensure face is clearly visible`);
      document.getElementById('regCapture').disabled = false;
      document.getElementById('regProgress').style.display = 'none';
      regDescriptors = [];
      return;
    }

    regDescriptors.push(detection.descriptor);

    // Update progress bar
    const pct = (((i + 1) / SAMPLES) * 100).toFixed(0);
    document.getElementById('regProgressFill').style.width  = pct + '%';
    document.getElementById('regProgressCount').textContent = `${i + 1}/${SAMPLES}`;

    // Brief pause between samples
    await new Promise(r => setTimeout(r, 300));
  }

  // Compute averaged descriptor
  const averaged = averageDescriptors(regDescriptors);
  regDescriptors  = [averaged];

  // Update UI
  document.getElementById('regCapture').disabled = false;
  document.getElementById('regFaceLabel').textContent = 'ENROLLED ✓';
  document.getElementById('regFaceLabel').className   = 'label-green';
  setRegStatus('success', `Face enrolled — ${SAMPLES} samples averaged`);
  showAlert('reg-alerts', 'success', 'Face biometrics captured successfully!');
}

/** Reset / clear the captured face data. */
function clearRegFace() {
  regDescriptors = [];
  document.getElementById('regFaceLabel').textContent = 'NOT ENROLLED';
  document.getElementById('regFaceLabel').className   = 'label-red';
  document.getElementById('regProgress').style.display = 'none';
  setRegStatus('idle', 'Face data cleared');
}

// ═══════════════════════════════════════════════════════════════
// REGISTRATION — SUBMIT
// ═══════════════════════════════════════════════════════════════

/** Validate inputs, hash password, and save the new user record. */
async function registerUser() {
  const username = document.getElementById('regUsername').value.trim();
  const password = document.getElementById('regPassword').value;
  const confirm  = document.getElementById('regConfirm').value;

  // --- Validation ---
  if (!username) {
    showAlert('reg-alerts', 'error', 'Username is required');
    return;
  }
  if (username.length < 3) {
    showAlert('reg-alerts', 'error', 'Username must be at least 3 characters');
    return;
  }
  if (db[username]) {
    showAlert('reg-alerts', 'error', 'Username already registered');
    return;
  }
  if (!password || password.length < 8) {
    showAlert('reg-alerts', 'error', 'Password must be at least 8 characters');
    return;
  }
  if (password !== confirm) {
    showAlert('reg-alerts', 'error', 'Passwords do not match');
    return;
  }

  // --- Persist ---
  const hash = await hashPassword(password);
  const hasFace = regDescriptors.length > 0;

  db[username] = {
    hash,
    face:    hasFace ? Array.from(regDescriptors[0]) : null,
    created: new Date().toISOString()
  };

  saveDB();

  showAlert(
    'reg-alerts',
    'success',
    `Account created${hasFace ? ' with face biometrics' : ' (no face enrolled)'}!`
  );

  // --- Reset form ---
  document.getElementById('regUsername').value = '';
  document.getElementById('regPassword').value = '';
  document.getElementById('regConfirm').value  = '';
  clearRegFace();
  stopRegCamera();

  // Redirect to login after a short delay
  setTimeout(() => showPage('login'), 1500);
}

// ═══════════════════════════════════════════════════════════════
// LOGIN — CAMERA
// ═══════════════════════════════════════════════════════════════

/** Request webcam access and start the login preview. */
async function startLoginCamera() {
  if (!modelsLoaded) {
    showAlert('login-alerts', 'warning', 'Biometric models still loading...');
    return;
  }

  try {
    loginStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' }
    });

    const video = document.getElementById('loginVideo');
    video.srcObject = loginStream;
    await video.play();

    document.getElementById('loginStartCam').textContent = '■ CAMERA ON';
    document.getElementById('loginVerify').disabled = false;
    setLoginStatus('scanning', 'Scanning for face...');
    startLoginDetection();

  } catch (err) {
    showAlert('login-alerts', 'error', 'Camera access denied: ' + err.message);
  }
}

/** Stop the login camera and clear the canvas. */
function stopLoginCamera() {
  if (loginStream) {
    loginStream.getTracks().forEach(t => t.stop());
    loginStream = null;
  }

  if (loginDetectionInterval) {
    clearInterval(loginDetectionInterval);
    loginDetectionInterval = null;
  }

  document.getElementById('loginStartCam').textContent = '📷 START CAMERA';
  document.getElementById('loginVerify').disabled = true;
  setLoginStatus('idle', 'Camera not started');

  const canvas = document.getElementById('loginCanvas');
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

/**
 * Run continuous face detection on the login video feed
 * and draw bounding box + landmark dots.
 */
function startLoginDetection() {
  const video  = document.getElementById('loginVideo');
  const canvas = document.getElementById('loginCanvas');

  loginDetectionInterval = setInterval(async () => {
    if (!video.videoWidth) return;

    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;

    const detection = await faceapi
      .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .catch(() => null);

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (detection) {
      const { x, y, width, height } = detection.detection.box;

      ctx.strokeStyle = '#00e5ff';
      ctx.lineWidth   = 2;
      ctx.strokeRect(x, y, width, height);

      ctx.fillStyle = 'rgba(0, 229, 255, 0.7)';
      detection.landmarks.positions.forEach(pt => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 1.5, 0, Math.PI * 2);
        ctx.fill();
      });

      ctx.fillStyle = '#00e5ff';
      ctx.font      = '11px Share Tech Mono';
      ctx.fillText('FACE IN FRAME', x, y - 8);

      setLoginStatus('detected', 'Face detected — click VERIFY & LOGIN');

    } else {
      setLoginStatus('scanning', 'No face detected — look at camera');
    }

  }, 200);
}

// ═══════════════════════════════════════════════════════════════
// LOGIN — VERIFY
// ═══════════════════════════════════════════════════════════════

/**
 * Verify the entered credentials and (if enrolled) the live face
 * against the stored biometric template.
 */
async function verifyLogin() {
  const username = document.getElementById('loginUsername').value.trim();
  const password = document.getElementById('loginPassword').value;

  // --- Basic credential checks ---
  if (!username || !password) {
    showAlert('login-alerts', 'error', 'Please enter username and password');
    return;
  }

  if (!db[username]) {
    showAlert('login-alerts', 'error', 'User not found');
    return;
  }

  const hash = await hashPassword(password);

  if (hash !== db[username].hash) {
    showAlert('login-alerts', 'error', 'Incorrect password');
    return;
  }

  // --- Password-only login (no face enrolled) ---
  if (!db[username].face) {
    showAlert(
      'login-alerts',
      'success',
      `✓ Password verified — Welcome back, ${username}! (No face enrollment on record)`
    );
    stopLoginCamera();
    return;
  }

  // --- Face verification required ---
  const video = document.getElementById('loginVideo');

  if (!loginStream || !video.videoWidth) {
    showAlert('login-alerts', 'error', 'Please start camera for face verification');
    return;
  }

  setLoginStatus('scanning', 'Analyzing face...');
  document.getElementById('loginVerify').disabled = true;

  const detection = await faceapi
    .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 416 }))
    .withFaceLandmarks()
    .withFaceDescriptor()
    .catch(() => null);

  if (!detection) {
    setLoginStatus('error', 'Face not detected — try again');
    showAlert('login-alerts', 'error', 'Could not detect face. Ensure good lighting and face the camera.');
    document.getElementById('loginVerify').disabled = false;
    return;
  }

  // Compute distance between live descriptor and stored template
  const storedDescriptor = new Float32Array(db[username].face);
  const distance = faceapi.euclideanDistance(detection.descriptor, storedDescriptor);

  console.log(`[BioAuth] Face verification distance: ${distance.toFixed(4)}`);

  if (distance < THRESHOLD) {
    const confidence = ((1 - distance) * 100).toFixed(1);
    setLoginStatus('success', `Identity confirmed — distance: ${distance.toFixed(3)}`);
    showAlert(
      'login-alerts',
      'success',
      `🔓 IDENTITY VERIFIED — Welcome, ${username}! (Confidence: ${confidence}%)`
    );
    stopLoginCamera();

    // Clear input fields after successful login
    document.getElementById('loginUsername').value = '';
    document.getElementById('loginPassword').value = '';

  } else {
    setLoginStatus('error', `Face mismatch — distance: ${distance.toFixed(3)}`);
    showAlert(
      'login-alerts',
      'error',
      `Face verification failed (score: ${distance.toFixed(3)} > threshold ${THRESHOLD}). Try better lighting.`
    );
    document.getElementById('loginVerify').disabled = false;
  }
}

// ═══════════════════════════════════════════════════════════════
// ENROLLED USERS LIST
// ═══════════════════════════════════════════════════════════════

/** Render the list of registered usernames on the login page. */
function updateUsersList() {
  const container = document.getElementById('usersList');
  const usernames = Object.keys(db);

  if (!usernames.length) {
    container.innerHTML = `
      <span style="font-family:Share Tech Mono;font-size:0.75rem;color:var(--muted)">
        No users registered yet
      </span>`;
    return;
  }

  container.innerHTML = usernames
    .map(u => `
      <div class="user-chip ${db[u].face ? 'has-face' : ''}">
        ${db[u].face ? '◉' : '○'} ${u} ${db[u].face ? '[+face]' : '[no face]'}
      </div>`)
    .join('');
}
