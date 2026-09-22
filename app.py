import logging
import os
import subprocess
from typing import Any, Dict
import uuid

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from video_processor import VideoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fitness Form Checker API",
    description="API и UI для анализа техники выполнения фитнес-упражнений",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: Dict[str, Dict[str, Any]] = {}
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def convert_to_h264(input_path: str, output_path: str) -> bool:
  """Конвертация видео в формат H.264 с помощью ffmpeg для воспроизведения в браузерах."""
  try:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-acodec",
        "aac",
        "-movflags",
        "+faststart",
        output_path,
    ]
    subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True
    )
    return True
  except Exception as e:
    logger.error(f"Ошибка FFmpeg конвертации: {str(e)}")
    return False


def run_video_processing(
    job_id: str,
    input_path: str,
    output_video_path: str,
    output_log_path: str,
    exercise_mode: str,
    sensitivity: float,
):
  """Фоновая задача обработки видеофайла."""
  try:
    JOBS[job_id]["status"] = "processing"
    logger.info(f"Начало обработки задачи {job_id}")

    mode_clean = exercise_mode.lower().strip()
    if "squat" in mode_clean or "присед" in mode_clean:
      normalized_mode = "squat"
    elif "pushup" in mode_clean or "отжим" in mode_clean:
      normalized_mode = "pushup"
    else:
      normalized_mode = "squat"

    raw_output_path = os.path.join(OUTPUT_DIR, f"{job_id}_raw_output.mp4")

    processor = VideoProcessor()
    processor.process_file(
        input_path=input_path,
        output_video_path=raw_output_path,
        output_path=raw_output_path,
        log_path=output_log_path,
        exercise_mode=normalized_mode,
        sensitivity=sensitivity,
    )

    if convert_to_h264(raw_output_path, output_video_path):
      if os.path.exists(raw_output_path):
        os.remove(raw_output_path)
    else:
      if os.path.exists(raw_output_path):
        os.rename(raw_output_path, output_video_path)

    JOBS[job_id]["status"] = "completed"
    JOBS[job_id]["processed_video_url"] = f"/api/v1/download/video/{job_id}"
    JOBS[job_id]["log_json_url"] = f"/api/v1/download/log/{job_id}"
    logger.info(f"Задача {job_id} успешно выполнена")

  except Exception as e:
    logger.error(
        f"Ошибка при обработке задачи {job_id}: {str(e)}", exc_info=True
    )
    JOBS[job_id]["status"] = "failed"
    JOBS[job_id]["error"] = str(e)


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
  return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fitness Form Checker</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            .card { border-radius: 15px; border: none; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
            .btn-primary { background-color: #4f46e5; border: none; padding: 10px 24px; border-radius: 8px; }
            .btn-primary:hover { background-color: #4338ca; }
            #loader, #resultsCard { display: none; }
        </style>
    </head>
    <body>
        <div class="container py-5" style="max-width: 800px;">
            <div class="text-center mb-4">
                <h1 class="fw-bold text-dark">🏋️‍♂️ Fitness Form Checker</h1>
                <p class="text-muted">Загрузите видео выполнения упражнения для автоматического анализа техники</p>
            </div>

            <div class="card p-4 mb-4">
                <form id="uploadForm">
                    <div class="mb-3">
                        <label class="form-label fw-bold">Выберите видеофайл (.mp4, .mov):</label>
                        <input type="file" id="videoFile" class="form-control" accept="video/*" required>
                    </div>

                    <div class="row mb-3">
                        <div class="col-md-6">
                            <label class="form-label fw-bold">Упражнение:</label>
                            <select id="exerciseMode" class="form-select">
                                <option value="squat">Приседания (Squats)</option>
                                <option value="pushup">Отжимания (Pushups)</option>
                            </select>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label fw-bold">Чувствительность (0.1 - 1.0):</label>
                            <input type="number" id="sensitivity" class="form-control" value="0.8" step="0.1" min="0.1" max="1.0">
                        </div>
                    </div>

                    <button type="submit" id="submitBtn" class="btn btn-primary w-100 fw-bold">🚀 Начать анализ</button>
                </form>
            </div>

            <div id="loader" class="text-center my-4">
                <div class="spinner-border text-primary" role="status" style="width: 3rem; height: 3rem;"></div>
                <p class="mt-2 text-secondary fw-semibold">Обработка видео нейросетью MediaPipe...</p>
            </div>

            <div id="resultsCard" class="card p-4">
                <h4 class="fw-bold text-success mb-2">✅ Анализ успешно завершен!</h4>
                
                <div class="alert alert-primary d-flex justify-content-between align-items-center my-3 p-3 rounded-3">
                    <span class="fw-bold fs-5">Средняя оценка техники (Average Score):</span>
                    <span id="avgScoreBadge" class="badge bg-success fs-4">0 / 100</span>
                </div>

                <div class="mb-3 text-center">
                    <video id="processedVideo" controls playsinline class="w-100 rounded shadow-sm" style="max-height: 450px; background: #000;"></video>
                </div>

                <div class="d-flex gap-2 mt-2">
                    <a id="downloadVideoBtn" class="btn btn-outline-primary flex-fill fw-semibold" download>📹 Скачать видео</a>
                    <a id="downloadLogBtn" class="btn btn-outline-secondary flex-fill fw-semibold" download>📊 Скачать JSON-лог</a>
                </div>
            </div>
        </div>

        <script>
            const uploadForm = document.getElementById('uploadForm');
            const loader = document.getElementById('loader');
            const resultsCard = document.getElementById('resultsCard');
            const submitBtn = document.getElementById('submitBtn');

            let currentJobId = null;
            let pollInterval = null;

            uploadForm.addEventListener('submit', async (e) => {
                e.preventDefault();

                const fileInput = document.getElementById('videoFile').files[0];
                const exerciseMode = document.getElementById('exerciseMode').value;
                const sensitivity = document.getElementById('sensitivity').value;

                if (!fileInput) return alert('Выберите файл!');

                const formData = new FormData();
                formData.append('file', fileInput);
                formData.append('exercise_mode', exerciseMode);
                formData.append('sensitivity', sensitivity);

                submitBtn.disabled = true;
                loader.style.display = 'block';
                resultsCard.style.display = 'none';

                try {
                    const response = await fetch('/api/v1/analyze', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) throw new Error('Ошибка при отправке файла');

                    const data = await response.json();
                    currentJobId = data.job_id;

                    pollInterval = setInterval(checkJobStatus, 1500);

                } catch (err) {
                    alert('Ошибка: ' + err.message);
                    loader.style.display = 'none';
                    submitBtn.disabled = false;
                }
            });

            async function checkJobStatus() {
                if (!currentJobId) return;

                try {
                    const res = await fetch(`/api/v1/jobs/${currentJobId}`);
                    const job = await res.json();

                    if (job.status === 'completed') {
                        clearInterval(pollInterval);
                        loader.style.display = 'none';
                        submitBtn.disabled = false;

                        const videoUrl = `/api/v1/download/video/${currentJobId}`;
                        const logUrl = `/api/v1/download/log/${currentJobId}`;

                        // Получение average_score из JSON-лога
                        try {
                            const logRes = await fetch(logUrl);
                            const logData = await logRes.json();
                            const avgScore = logData.summary ? logData.summary.average_score : 0;
                            document.getElementById('avgScoreBadge').innerText = `${avgScore} / 100`;
                        } catch (e) {
                            console.error("Не удалось прочитать average_score:", e);
                        }

                        const processedVideo = document.getElementById('processedVideo');
                        processedVideo.src = videoUrl;
                        processedVideo.load();

                        document.getElementById('downloadVideoBtn').href = videoUrl;
                        document.getElementById('downloadLogBtn').href = logUrl;

                        resultsCard.style.display = 'block';
                    } else if (job.status === 'failed') {
                        clearInterval(pollInterval);
                        loader.style.display = 'none';
                        submitBtn.disabled = false;
                        alert('Ошибка обработки: ' + (job.error || 'Неизвестная ошибка'));
                    }
                } catch (err) {
                    console.error('Ошибка проверки статуса:', err);
                }
            }
        </script>
    </body>
    </html>
    """


@app.post("/api/v1/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    exercise_mode: str = Form("squat"),
    sensitivity: float = Form(0.8),
):
  job_id = str(uuid.uuid4())
  input_path = os.path.join(OUTPUT_DIR, f"{job_id}_input.mp4")
  output_video_path = os.path.join(OUTPUT_DIR, f"{job_id}_output.mp4")
  output_log_path = os.path.join(OUTPUT_DIR, f"{job_id}_log.json")

  with open(input_path, "wb") as buffer:
    buffer.write(await file.read())

  JOBS[job_id] = {
      "job_id": job_id,
      "status": "pending",
      "processed_video_url": None,
      "log_json_url": None,
      "error": None,
  }

  background_tasks.add_task(
      run_video_processing,
      job_id,
      input_path,
      output_video_path,
      output_log_path,
      exercise_mode,
      sensitivity,
  )

  return {"job_id": job_id, "status": "pending"}


@app.get("/api/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
  if job_id not in JOBS:
    raise HTTPException(status_code=404, detail="Job not found")
  return JOBS[job_id]


@app.get("/api/v1/download/video/{job_id}")
async def download_video(job_id: str):
  video_path = os.path.join(OUTPUT_DIR, f"{job_id}_output.mp4")
  if not os.path.exists(video_path):
    raise HTTPException(status_code=404, detail="Video file not found")

  return FileResponse(
      video_path,
      media_type="video/mp4",
      filename=f"processed_{job_id}.mp4",
      headers={"Accept-Ranges": "bytes"},
  )


@app.get("/api/v1/download/log/{job_id}")
async def download_log(job_id: str):
  log_path = os.path.join(OUTPUT_DIR, f"{job_id}_log.json")
  if not os.path.exists(log_path):
    raise HTTPException(status_code=404, detail="Log file not found")

  return FileResponse(
      log_path, media_type="application/json", filename=f"log_{job_id}.json"
  )