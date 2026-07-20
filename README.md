# <강조> 모든 .py파일은 가상환경에서 실행하고, main.py를 작동시킨후, ngrok 터널링을 진행한 후에 크롬/파이어폭스 창에 "appointee-dreary-unisexual.ngrok-free.dev" 입력후 실행

ngrok config add-authtoken 복사하신_인증_토큰 >> 터널링 하기위한 토큰 등록
ngrok http --domain=appointee-dreary-unisexual.ngrok-free.dev 사용중인_포트번호 >> ngrok 터널링

1. del_user : 사용자 명단 삭제 시스템
2. main.py : 하나의 서버 역할
3. index.html : 로그인하여 인원을 등록 및 개별 얼굴 데이터셋을 확보 하기 위한 웹페이지
4. detect_realsys.py : 데이터베이스를 바탕으로 카메라를 통해 사람의 얼굴 객체를 식별해 해당 사람이 데이터베이스에 존재하는 사람인지 있다면 그사람의 그룹이 어딘지에 따른 표식 부여
