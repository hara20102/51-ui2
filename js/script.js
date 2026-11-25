import { FilesetResolver, HandLandmarker, PoseLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/vision_bundle.mjs';

const video = document.querySelector('#webcam');
const personCanvas = document.querySelector('#personCanvas');
const personContext = personCanvas.getContext('2d');
const particleCanvas = document.querySelector('#particleCanvas');
const particleContext = particleCanvas.getContext('2d');
const maskCanvas = document.createElement('canvas');
const maskContext = maskCanvas.getContext('2d');
const eraseCanvas = document.createElement('canvas');
const eraseContext = eraseCanvas.getContext('2d');
const stage = document.querySelector('#cameraStage');
const cameraCard = document.querySelector('.camera-card');
const cameraButton = document.querySelector('#cameraButton');
const cameraButtonText = document.querySelector('#cameraButtonText');
const fullscreenButton = document.querySelector('#fullscreenButton');
const fullscreenButtonText = document.querySelector('#fullscreenButtonText');
const statusPill = document.querySelector('.status-pill');
const statusText = document.querySelector('#statusText');
const statusDot = document.querySelector('#statusDot');
const detectionText = document.querySelector('#detectionText');
const emptyState = document.querySelector('#emptyState');
const loadingState = document.querySelector('#loadingState');
const noticeText = document.querySelector('#noticeText');

let handLandmarker = null;
let poseLandmarker = null;
let visionFileset = null;
let cameraStream = null;
let animationFrameId = null;
let isDetecting = false;
let lastVideoTime = -1;
let poseErrorShown = false;
let previousMaskData = null;
let particleFrame = 0;
const particleGlyphs = '亂碼錯誤資料異常混沌失序破碎虛構殘影無序重組錯位字符空白雜訊';
const gestureMarks = [];
const previousPalmPositions = new Map();
const gestureMarkLifetime = 1000;
const mirrorPreviewEnabled = true;
let okGestureStartedAt = null;
let okGestureTriggered = false;

function setStatus(message, state = '') {
    statusText.textContent = message;
    statusPill.dataset.state = state;
}

function updateFullscreenButton() {
    const isFullscreen = document.fullscreenElement === stage;
    fullscreenButtonText.textContent = isFullscreen ? '退出全螢幕' : '畫面全螢幕';
    fullscreenButton.setAttribute('aria-pressed', String(isFullscreen));
}

function distanceBetweenLandmarks(first, second, width, height) {
    return Math.hypot((first.x - second.x) * width, (first.y - second.y) * height);
}

function isOkGesture(landmarks, width, height) {
    const palmSize = distanceBetweenLandmarks(landmarks[5], landmarks[17], width, height);
    const thumbIndexDistance = distanceBetweenLandmarks(landmarks[4], landmarks[8], width, height);
    if (thumbIndexDistance > Math.max(14, palmSize * 0.3)) return false;

    // 中指、無名指與小指伸直時，指尖會比 PIP 關節離手腕更遠。
    return [[12, 10], [16, 14], [20, 18]].every(([tipIndex, pipIndex]) => {
        const tipDistance = distanceBetweenLandmarks(landmarks[tipIndex], landmarks[0], width, height);
        const pipDistance = distanceBetweenLandmarks(landmarks[pipIndex], landmarks[0], width, height);
        return tipDistance > pipDistance * 1.12;
    });
}

function updateRestartGesture(hands, width, height, now) {
    const isHoldingOk = hands.some((landmarks) => isOkGesture(landmarks, width, height));
    if (!isHoldingOk) {
        okGestureStartedAt = null;
        okGestureTriggered = false;
        return;
    }

    if (okGestureTriggered) {
        detectionText.textContent = '重新開始完成，放開 OK 手勢即可再次使用';
        return;
    }

    okGestureStartedAt ??= now;
    const elapsed = now - okGestureStartedAt;
    const remaining = Math.max(0, (1500 - elapsed) / 1000);
    detectionText.textContent = `OK 手勢：維持 ${remaining.toFixed(1)} 秒以重新開始`;

    if (elapsed >= 1500) {
        gestureMarks.length = 0;
        previousPalmPositions.clear();
        eraseContext.clearRect(0, 0, eraseCanvas.width, eraseCanvas.height);
        particleContext.clearRect(0, 0, particleCanvas.width, particleCanvas.height);
        okGestureTriggered = true;
        detectionText.textContent = '重新開始完成，放開 OK 手勢即可再次使用';
    }
}

async function toggleFullscreen() {
    try {
        if (document.fullscreenElement === stage) {
            await document.exitFullscreen();
        } else if (stage.requestFullscreen) {
            await stage.requestFullscreen();
        } else {
            noticeText.textContent = '此瀏覽器不支援全螢幕功能。';
        }
    } catch (error) {
        noticeText.textContent = `無法切換全螢幕：${error.message || '請確認瀏覽器允許此操作。'}`;
    }
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
    gestureMarks.length = 0;
    previousPalmPositions.clear();
    okGestureStartedAt = null;
    okGestureTriggered = false;
    eraseContext.clearRect(0, 0, eraseCanvas.width, eraseCanvas.height);
    stage.dataset.active = 'false';
    cameraCard.dataset.active = 'false';
    emptyState.hidden = false;
    loadingState.hidden = true;
    cameraButtonText.textContent = '開啟相機';
    detectionText.textContent = '尚未偵測人物';
    personContext.clearRect(0, 0, personCanvas.width, personCanvas.height);
    particleContext.clearRect(0, 0, particleCanvas.width, particleCanvas.height);
    setStatus('相機尚未啟動');
}

function drawHandOverlay(handResult, poseResult) {
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return;

    const hands = handResult.landmarks ?? [];
    const poses = poseResult.landmarks ?? [];
    updateGestureBrush(hands, width, height, performance.now());
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

function updateGestureBrush(hands, width, height, now) {
    const activeHands = new Set();

    hands.forEach((landmarks, handIndex) => {
        // 以手腕與四個掌根關鍵點的平均位置代表掌心。
        const palmIndices = [0, 5, 9, 13, 17];
        const palm = palmIndices.reduce((center, index) => ({
            x: center.x + landmarks[index].x / palmIndices.length,
            y: center.y + landmarks[index].y / palmIndices.length
        }), { x: 0, y: 0 });
        const palmSize = Math.hypot(
            (landmarks[5].x - landmarks[17].x) * width,
            (landmarks[5].y - landmarks[17].y) * height
        );
        const key = `hand-${handIndex}`;
        activeHands.add(key);
        const previous = previousPalmPositions.get(key);

        if (previous) {
            const dx = (palm.x - previous.x) * width;
            const dy = (palm.y - previous.y) * height;
            const distance = Math.hypot(dx, dy);
            const elapsed = Math.max(1, now - previous.time);
            const speed = distance / elapsed;

            // 揮動超過速度門檻時，沿著掌心位移補點，形成連續筆跡。
            if (speed > 0.12 && distance > 1.5) {
                const count = Math.min(12, Math.max(1, Math.ceil(distance / 14)));
                for (let step = 1; step <= count; step += 1) {
                    const progress = step / count;
                    const centerX = (previous.x + (palm.x - previous.x) * progress) * width;
                    const centerY = (previous.y + (palm.y - previous.y) * progress) * height;
                    const brushRadius = Math.max(34, palmSize * 1.15);
                    const glyphCount = Math.max(1, Math.min(2, Math.ceil(palmSize / 40)));

                    // 每個筆跡點以掌心寬度為範圍，鋪滿一團隨機字元。
                    for (let glyphIndex = 0; glyphIndex < glyphCount; glyphIndex += 1) {
                        const angle = Math.random() * Math.PI * 2;
                        // 半徑採中心偏重的分布，越靠外圍的字元越少且距離更鬆散。
                        const radius = Math.pow(Math.random(), 1.8) * brushRadius;
                        const mark = {
                            x: centerX + Math.cos(angle) * radius,
                            y: centerY + Math.sin(angle) * radius,
                            time: now,
                            glyph: particleGlyphs[Math.floor(Math.random() * particleGlyphs.length)],
                            size: 28 + Math.random() * 8,
                            rotation: (Math.random() - 0.5) * 0.7,
                            color: ['#c2ff7b', '#55e7da', '#eef8df'][Math.floor(Math.random() * 3)]
                        };
                        gestureMarks.push(mark);
                        // 擦除遮罩以多個鄰近字形加密，視覺字元仍維持稀疏。
                        for (let eraseIndex = 0; eraseIndex < 3; eraseIndex += 1) {
                            const scatterX = eraseIndex === 0 ? 0 : (Math.random() - 0.5) * mark.size * 1.4;
                            const scatterY = eraseIndex === 0 ? 0 : (Math.random() - 0.5) * mark.size * 1.4;
                            eraseContext.save();
                            eraseContext.fillStyle = '#000';
                            eraseContext.filter = 'blur(3px)';
                            eraseContext.font = `700 ${mark.size}px monospace`;
                            eraseContext.textAlign = 'center';
                            eraseContext.textBaseline = 'middle';
                            eraseContext.translate(mark.x + scatterX, mark.y + scatterY);
                            eraseContext.rotate(mark.rotation);
                            eraseContext.fillText(mark.glyph, 0, 0);
                            eraseContext.restore();
                        }
                    }
                }
            }
        }
        previousPalmPositions.set(key, { x: palm.x, y: palm.y, time: now });
    });

    for (const key of previousPalmPositions.keys()) {
        if (!activeHands.has(key)) previousPalmPositions.delete(key);
    }
}

function drawGestureMarks(width, height, now) {
    for (let index = gestureMarks.length - 1; index >= 0; index -= 1) {
        const mark = gestureMarks[index];
        const progress = (now - mark.time) / gestureMarkLifetime;
        if (progress >= 1) {
            gestureMarks.splice(index, 1);
            continue;
        }

        particleContext.save();
        particleContext.globalAlpha = Math.max(0, 1 - progress);
        particleContext.fillStyle = mark.color;
        particleContext.font = `700 ${mark.size}px monospace`;
        particleContext.textAlign = 'center';
        particleContext.textBaseline = 'middle';
        particleContext.translate(mark.x, mark.y);
        particleContext.rotate(mark.rotation);
        if (mirrorPreviewEnabled) particleContext.scale(-1, 1);
        particleContext.fillText(mark.glyph, 0, 0);
        particleContext.restore();
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
    if (particleCanvas.width !== width || particleCanvas.height !== height) {
        particleCanvas.width = width;
        particleCanvas.height = height;
    }
    if (eraseCanvas.width !== width || eraseCanvas.height !== height) {
        // 影像尺寸變更時同步重建永久擦除圖層。
        eraseCanvas.width = width;
        eraseCanvas.height = height;
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
    personContext.filter = 'blur(0.35px)';
    personContext.drawImage(maskCanvas, 0, 0, width, height);
    personContext.filter = 'none';
    personContext.globalCompositeOperation = 'source-over';

    // 累積擦除遮罩不隨字元淡出而清除，人物被筆跡擦過後會持續消失。
    personContext.globalCompositeOperation = 'destination-out';
    personContext.drawImage(eraseCanvas, 0, 0, width, height);
    personContext.globalCompositeOperation = 'source-over';

    drawLetterParticles(maskData, maskWidth, maskHeight, width, height);
}

function drawLetterParticles(maskData, maskWidth, maskHeight, width, height) {
    particleContext.clearRect(0, 0, width, height);
    particleFrame += 1;
    particleContext.textAlign = 'center';
    particleContext.textBaseline = 'middle';

    // 沿著遮罩的明暗交界取樣，將字元散落在輪廓外側與邊緣。
    const step = Math.max(2, Math.round(Math.min(maskWidth, maskHeight) / 100));
    for (let y = 2; y < maskHeight - 2; y += step) {
        for (let x = 2; x < maskWidth - 2; x += step) {
            const index = y * maskWidth + x;
            const alpha = maskData[index];
            const horizontal = maskData[index - 2] - maskData[index + 2];
            const vertical = maskData[index - maskWidth * 2] - maskData[index + maskWidth * 2];
            const edgeStrength = Math.hypot(horizontal, vertical);
            if (alpha < 0.12 || alpha > 0.9 || edgeStrength < 0.16) continue;

            // 固定位置雜湊讓字元不閃爍，只保留輕微的逐幀漂移感。
            const seed = (x * 73856093 ^ y * 19349663) >>> 0;
            if (seed % 100 > 44) continue;
            const directionX = horizontal / (edgeStrength || 1);
            const directionY = vertical / (edgeStrength || 1);
            const drift = ((particleFrame + seed) % 9) - 4;
            const scaleX = width / maskWidth;
            const scaleY = height / maskHeight;
            const px = (x + directionX * (2 + drift * 0.35)) * scaleX;
            const py = (y + directionY * (2 + drift * 0.35)) * scaleY;
            const glyph = particleGlyphs[seed % particleGlyphs.length];
            const size = 24 + (seed % 9);

            particleContext.globalAlpha = 0.42 + (seed % 45) / 100;
            particleContext.fillStyle = seed % 3 === 0 ? '#c2ff7b' : seed % 3 === 1 ? '#55e7da' : '#eef8df';
            particleContext.font = `600 ${size}px monospace`;
            if (mirrorPreviewEnabled) {
                particleContext.save();
                particleContext.translate(px, py);
                particleContext.scale(-1, 1);
                particleContext.fillText(glyph, 0, 0);
                particleContext.restore();
            } else {
                particleContext.fillText(glyph, px, py);
            }
        }
    }
    particleContext.globalAlpha = 1;
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
                if (eraseCanvas.width !== video.videoWidth || eraseCanvas.height !== video.videoHeight) {
                    eraseCanvas.width = video.videoWidth;
                    eraseCanvas.height = video.videoHeight;
                }
                if (particleCanvas.width !== video.videoWidth || particleCanvas.height !== video.videoHeight) {
                    particleCanvas.width = video.videoWidth;
                    particleCanvas.height = video.videoHeight;
                }
                drawHandOverlay(handResult, poseResult);
                updateRestartGesture(handResult.landmarks ?? [], video.videoWidth, video.videoHeight, timestamp);
                if (mask) {
                    drawPersonCutout(mask);
                } else {
                    previousMaskData = null;
                    personContext.clearRect(0, 0, personCanvas.width, personCanvas.height);
                    particleContext.clearRect(0, 0, particleCanvas.width, particleCanvas.height);
                }
                drawGestureMarks(video.videoWidth, video.videoHeight, timestamp);
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

fullscreenButton.addEventListener('click', toggleFullscreen);
document.addEventListener('fullscreenchange', updateFullscreenButton);

