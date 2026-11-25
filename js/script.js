import { FilesetResolver, HandLandmarker, PoseLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/vision_bundle.mjs';

const video = document.querySelector('#webcam');
const canvas = document.querySelector('#landmarkCanvas');
const drawingContext = canvas.getContext('2d');
const personCanvas = document.querySelector('#personCanvas');
const personContext = personCanvas.getContext('2d');
const maskCanvas = document.createElement('canvas');
const maskContext = maskCanvas.getContext('2d');
const stage = document.querySelector('#cameraStage');
const cameraCard = document.querySelector('.camera-card');
const cameraButton = document.querySelector('#cameraButton');
const cameraButtonText = document.querySelector('#cameraButtonText');
const mirrorToggle = document.querySelector('#mirrorToggle');
const backgroundToggle = document.querySelector('#backgroundToggle');
const mirrorBadge = document.querySelector('#mirrorBadge');
const mirrorDescription = document.querySelector('#mirrorDescription');
const statusPill = document.querySelector('.status-pill');
const statusText = document.querySelector('#statusText');
const statusDot = document.querySelector('#statusDot');
const detectionText = document.querySelector('#detectionText');
const emptyState = document.querySelector('#emptyState');
const loadingState = document.querySelector('#loadingState');
const noticeText = document.querySelector('#noticeText');

// MediaPipe 手部骨架的標準連線，共 21 個節點。
const handConnections = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]
];

let handLandmarker = null;
let poseLandmarker = null;
let visionFileset = null;
let cameraStream = null;
let animationFrameId = null;
let isDetecting = false;
let lastVideoTime = -1;
let poseErrorShown = false;
let previousMaskData = null;

function setStatus(message, state = '') {
    statusText.textContent = message;
    statusPill.dataset.state = state;
}

function updateMirrorMode() {
    const isMirrored = mirrorToggle.checked;
    stage.dataset.mirror = String(isMirrored);
    mirrorBadge.textContent = isMirrored ? '鏡像模式' : '原始方向';
    mirrorBadge.classList.toggle('camera-badge-off', !isMirrored);
    mirrorDescription.textContent = isMirrored
        ? '左右翻轉畫面，像照鏡子一樣'
        : '顯示攝影機拍到的原始左右方向';
}

function updateBackgroundRemoval() {
    const isRemovingBackground = backgroundToggle.checked;
    stage.dataset.backgroundRemoval = String(isRemovingBackground);
}

async function createHandLandmarker() {
    return HandLandmarker.createFromOptions(visionFileset, {
        baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'GPU'
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
}

async function createPoseLandmarker() {
    return PoseLandmarker.createFromOptions(visionFileset, {
        baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
            delegate: 'GPU'
        },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
        outputSegmentationMasks: true
    });
}

async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
        setStatus('瀏覽器不支援相機', 'error');
        noticeText.textContent = '此瀏覽器無法存取相機，請改用最新版 Chrome、Edge 或 Safari，並透過 localhost 或 HTTPS 開啟。';
        return;
    }

    cameraButton.disabled = true;
    setStatus('正在啟動相機…', 'loading');
    loadingState.hidden = false;
    noticeText.textContent = '請在瀏覽器提示中允許使用相機。';

    try {
        // 先取得相機權限，預設選擇使用者面向的前鏡頭。
        cameraStream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 800 } }
        });
        video.srcObject = cameraStream;
        await video.play();

        if (!handLandmarker || !poseLandmarker) {
            setStatus('正在載入人物與手部模型…', 'loading');
            visionFileset ??= await FilesetResolver.forVisionTasks(
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm'
            );
            handLandmarker ??= await createHandLandmarker();
            poseLandmarker ??= await createPoseLandmarker();
        }

        stage.dataset.active = 'true';
        cameraCard.dataset.active = 'true';
        emptyState.hidden = true;
        loadingState.hidden = true;
        cameraButtonText.textContent = '關閉相機';
        detectionText.textContent = '請站在鏡頭前開始偵測';
        setStatus('相機已啟動，等待人物', 'ready');
        noticeText.textContent = '提示：請讓人物完整進入畫面，以便進行去背與偵測。';
        isDetecting = true;
        poseErrorShown = false;
        detectFrame();
    } catch (error) {
        stopCamera();
        setStatus('無法啟動相機', 'error');
        noticeText.textContent = explainCameraError(error);
    } finally {
        cameraButton.disabled = false;
        loadingState.hidden = true;
    }
}

function explainCameraError(error) {
    if (error?.name === 'NotAllowedError' || error?.name === 'PermissionDeniedError') {
        return '相機權限未開啟。請在瀏覽器網址列的網站設定中允許相機，再重新嘗試。';
    }
    if (error?.name === 'NotFoundError' || error?.name === 'DevicesNotFoundError') {
        return '找不到可用的相機，請確認 webcam 已連接後重試。';
    }
    if (error?.name === 'NotReadableError' || error?.name === 'TrackStartError') {
        return '相機目前無法使用，可能正被其他應用程式佔用。請關閉其他正在使用相機的程式後重試。';
    }
    const detail = error?.message ? `（${error.message}）` : '';
    return `模型載入或相機啟動失敗${detail}。請確認網路可載入 MediaPipe CDN，並透過 localhost 或 HTTPS 開啟此頁面。`;
}

function stopCamera() {
    isDetecting = false;
    if (animationFrameId !== null) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    cameraStream?.getTracks().forEach((track) => track.stop());
    cameraStream = null;
    video.srcObject = null;
    lastVideoTime = -1;
    previousMaskData = null;
    stage.dataset.active = 'false';
    cameraCard.dataset.active = 'false';
    emptyState.hidden = false;
    loadingState.hidden = true;
    cameraButtonText.textContent = '開啟相機';
    detectionText.textContent = '尚未偵測人物';
    drawingContext.clearRect(0, 0, canvas.width, canvas.height);
    personContext.clearRect(0, 0, personCanvas.width, personCanvas.height);
    setStatus('相機尚未啟動');
}

function drawConnections(landmarks, connections, color, lineWidth, width, height) {
    drawingContext.lineWidth = lineWidth;
    drawingContext.lineCap = 'round';
    drawingContext.strokeStyle = color;
    connections.forEach(([startIndex, endIndex]) => {
        const start = landmarks[startIndex];
        const end = landmarks[endIndex];
        if (!start || !end || (start.visibility ?? 1) < 0.25 || (end.visibility ?? 1) < 0.25) return;
        drawingContext.beginPath();
        drawingContext.moveTo(start.x * width, start.y * height);
        drawingContext.lineTo(end.x * width, end.y * height);
        drawingContext.stroke();
    });
}

function drawHandLandmarks(hands, width, height) {
    hands.forEach((landmarks) => {
        drawConnections(landmarks, handConnections, '#c2ff7b', Math.max(2, width * 0.003), width, height);
        landmarks.forEach((landmark, index) => {
            const radius = index === 0 ? Math.max(5, width * 0.008) : Math.max(3.5, width * 0.0055);
            drawingContext.beginPath();
            drawingContext.arc(landmark.x * width, landmark.y * height, radius, 0, Math.PI * 2);
            drawingContext.fillStyle = index === 0 ? '#c2ff7b' : '#55e7da';
            drawingContext.fill();
            drawingContext.lineWidth = 1.5;
            drawingContext.strokeStyle = '#10201e';
            drawingContext.stroke();
        });
    });
}

function drawHandOverlay(handResult, poseResult) {
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return;

    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
    }
    drawingContext.clearRect(0, 0, width, height);

    const hands = handResult.landmarks ?? [];
    const poses = poseResult.landmarks ?? [];
    drawHandLandmarks(hands, width, height);

    if (poses.length > 0 && hands.length > 0) {
        detectionText.textContent = `偵測到人物與 ${hands.length} 隻手`;
        setStatus('人物與手部偵測中', 'ready');
    } else if (poses.length > 0) {
        detectionText.textContent = '已偵測人物';
        setStatus('人物偵測中', 'ready');
    } else if (hands.length > 0) {
        detectionText.textContent = `偵測到 ${hands.length} 隻手`;
        setStatus('手部偵測中，等待人物', 'ready');
    } else {
        detectionText.textContent = '等待偵測人物與手部';
        setStatus('相機已啟動，等待人物', 'ready');
    }
}

function drawPersonCutout(mask) {
    const maskWidth = mask.width;
    const maskHeight = mask.height;
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!maskWidth || !maskHeight || !width || !height) return;

    if (maskCanvas.width !== maskWidth || maskCanvas.height !== maskHeight) {
        maskCanvas.width = maskWidth;
        maskCanvas.height = maskHeight;
    }
    if (personCanvas.width !== width || personCanvas.height !== height) {
        personCanvas.width = width;
        personCanvas.height = height;
    }

    const maskData = mask.getAsFloat32Array();
    if (maskData.length !== maskWidth * maskHeight) {
        throw new Error('人物遮罩尺寸與資料長度不一致');
    }

    // 以前後兩幀遮罩進行時間平滑，減少人物邊緣來回跳動。
    if (!previousMaskData || previousMaskData.length !== maskData.length) {
        previousMaskData = new Float32Array(maskData.length);
    }

    const pixels = maskContext.createImageData(maskWidth, maskHeight);
    for (let index = 0; index < maskData.length; index += 1) {
        const colorIndex = index * 4;
        const currentAlpha = Math.max(0, Math.min(1, maskData[index]));
        const previousAlpha = previousMaskData[index];
        const smoothedAlpha = previousAlpha * 0.72 + currentAlpha * 0.28;
        previousMaskData[index] = smoothedAlpha;

        let stableAlpha = smoothedAlpha;
        if (stableAlpha < 0.08) stableAlpha = 0;
        if (stableAlpha > 0.92) stableAlpha = 1;

        const alpha = Math.round(stableAlpha * 255);
        pixels.data[colorIndex] = 255;
        pixels.data[colorIndex + 1] = 255;
        pixels.data[colorIndex + 2] = 255;
        pixels.data[colorIndex + 3] = alpha;
    }
    maskContext.putImageData(pixels, 0, 0);

    personContext.clearRect(0, 0, width, height);
    personContext.globalCompositeOperation = 'source-over';
    personContext.drawImage(video, 0, 0, width, height);
    personContext.globalCompositeOperation = 'destination-in';
    personContext.filter = 'blur(1px)';
    personContext.drawImage(maskCanvas, 0, 0, width, height);
    personContext.filter = 'none';
    personContext.globalCompositeOperation = 'source-over';
}

function detectFrame() {
    if (!isDetecting) return;

    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const timestamp = performance.now();
        try {
            const handResult = handLandmarker.detectForVideo(video, timestamp);
            const poseResult = poseLandmarker.detectForVideo(video, timestamp);
            const mask = poseResult.segmentationMasks?.[0];
            try {
                drawHandOverlay(handResult, poseResult);
                if (mask) {
                    if (backgroundToggle.checked) {
                        drawPersonCutout(mask);
                    } else {
                        personContext.clearRect(0, 0, personCanvas.width, personCanvas.height);
                    }
                } else {
                    previousMaskData = null;
                    personContext.clearRect(0, 0, personCanvas.width, personCanvas.height);
                }
            } finally {
                mask?.close();
            }
        } catch (error) {
            if (!poseErrorShown) {
                poseErrorShown = true;
                noticeText.textContent = `即時偵測暫時無法處理：${error.message || '未知錯誤'}`;
            }
        }
    }
    animationFrameId = requestAnimationFrame(detectFrame);
}

cameraButton.addEventListener('click', () => {
    if (cameraStream) {
        stopCamera();
        noticeText.textContent = '相機已關閉。';
    } else {
        startCamera();
    }
});

mirrorToggle.addEventListener('change', updateMirrorMode);
backgroundToggle.addEventListener('change', () => {
    updateBackgroundRemoval();
    if (!backgroundToggle.checked) {
        personContext.clearRect(0, 0, personCanvas.width, personCanvas.height);
    }
});
updateMirrorMode();
updateBackgroundRemoval();
