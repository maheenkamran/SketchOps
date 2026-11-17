# Build from project root:  docker build -f sandbox/Dockerfile -t sketchops-sandbox .
FROM mcr.microsoft.com/playwright/python:v1.49.0-jammy
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.163.1 uvicorn==0.54.0 playwright==1.49.0
RUN mkdir -p /app/vendor && python -c "import urllib.request as u; u.urlretrieve('https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4.3.5','/app/vendor/tailwind.js')"
COPY sandbox/render.py sandbox/server.py /app/
USER pwuser
EXPOSE 9000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000"]
