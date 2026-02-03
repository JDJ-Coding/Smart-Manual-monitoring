my_manual_project/
├── app.py                # 제가 작성해 드린 전체 코드 파일
├── manuals/              # 관리 중인 모든 PDF 매뉴얼을 넣는 곳 (직접 생성)
│   ├── inverter_A.pdf
│   ├── valve_B.pdf
│   └── ...
├── manual_db/            # (자동 생성) AI가 학습한 수치 데이터가 저장되는 곳
│   ├── index.faiss
│   └── index.pkl
└── requirements.txt      # 설치가 필요한 라이브러리 목록