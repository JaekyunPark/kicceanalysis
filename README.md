# Survey Analysis Project

설문조사 데이터 분석을 위한 Django 웹 애플리케이션

## 프로젝트 구조

```
survey_project/
├── survey_project/          # 프로젝트 설정 디렉토리
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── surveys/                 # 메인 앱
│   ├── migrations/
│   ├── templates/
│   │   └── surveys/
│   │       ├── base.html
│   │       ├── home.html
│   │       ├── data_upload.html
│   │       ├── codebook_upload.html
│   │       ├── data_list.html
│   │       └── codebook_list.html
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── forms.py
│   ├── views.py
│   └── urls.py
├── analysis/                # 분석 앱
│   ├── migrations/
│   ├── templates/
│   │   └── analysis/
│   │       ├── unified_analysis.html
│   │       └── analysis_result.html
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── views.py
│   └── urls.py
├── manage.py
└── requirements.txt
```

## 설치 방법

### 1. 가상환경 생성 및 활성화
```bash
# Windows
python -m venv myenv
myenv\Scripts\activate

# Mac/Linux
python3 -m venv myenv
source myenv/bin/activate
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. 데이터베이스 마이그레이션
```bash
python manage.py makemigrations
python manage.py migrate
```

### 4. 슈퍼유저 생성
```bash
python manage.py createsuperuser
```

### 5. 서버 실행
```bash
python manage.py runserver
```

### 6. 접속
브라우저에서 http://127.0.0.1:8000/ 접속

## 주요 기능

### 1. 데이터 관리
- CSV/Excel 파일 업로드
- 데이터셋 목록 조회 및 관리

### 2. 코드북 관리
- Excel 형식 코드북 업로드
- 변수 레이블 및 값 레이블 관리
- 데이터셋과 코드북 연결

### 3. 통합 분석
- 교차분석, 빈도분석, 평균분석 통합
- 다양한 통계량 선택 (빈도, 백분율, 평균, 표준편차, 4분위수 등)
- 자동 통계 검정 (Chi-square, t-test, ANOVA)
- 전문적인 표 형식 출력

## 코드북 형식

Excel 파일에 다음 컬럼이 필요합니다:

| Variable | Variable Label | Value | Value Label |
|----------|---------------|-------|-------------|
| Q1       | 성별          | 1     | 남성        |
| Q1       |               | 2     | 여성        |
| Q2       | 연령대        | 1     | 20대        |
| Q2       |               | 2     | 30대        |

## 문의사항

문제가 발생하면 이슈를 등록해주세요.
