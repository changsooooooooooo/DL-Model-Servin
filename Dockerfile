FROM python:3.8-bullseye

# 환경변수
ENV LIBRARY_PATH=/lib:/usr/lib

# 호스트와 연결할 포트
EXPOSE 7500

# cp
COPY . .

# 패키지 설치
RUN pip install -r requirements.txt

# cd
WORKDIR /was

# 실행 명령어
CMD ["python", "app.py"]
