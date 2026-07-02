import sqlite3
import os

DB_PATH = "security_system.db"

def delete_user_by_name(target_name):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # ⭐️ 핵심 변경: '=' 대신 'LIKE'를 쓰고, 이름 뒤에 '%' 기호를 붙여서 
        # "임형권"으로 시작하는 모든 데이터(_정면, _좌측면 등)를 찾게 합니다.
        search_pattern = f"{target_name}%" 
        cursor.execute("SELECT name, image_path FROM users WHERE name LIKE ?", (search_pattern,))
        rows = cursor.fetchall()

        if not rows:
            print(f"❌ DB에 '{target_name}' 이(가) 포함된 데이터가 없습니다.")
            
            # ✨ 디버깅 도우미: 현재 DB에 등록된 실제 이름들을 보여줍니다.
            cursor.execute("SELECT DISTINCT name FROM users")
            existing_names = [row[0] for row in cursor.fetchall()]
            print(f"👉 현재 DB에 등록된 실제 이름 목록: {existing_names}")
            
            conn.close()
            return

        # 2. 이미지 파일 삭제
        for row in rows:
            db_name = row[0]
            file_path = row[1]
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ 파일 삭제 완료: {file_path} ({db_name})")
            else:
                print(f"⚠️ 이미 지워진 파일입니다: {file_path}")

        # 3. DB 기록 삭제 (마찬가지로 LIKE 적용)
        cursor.execute("DELETE FROM users WHERE name LIKE ?", (search_pattern,))
        deleted_count = cursor.rowcount  # 몇 개가 지워졌는지 셉니다.
        conn.commit()
        conn.close()

        print(f"✅ '{target_name}' 님의 모든 데이터({deleted_count}건)가 완벽하게 삭제되었습니다.")

    except Exception as e:
        print(f"❌ 에러 발생: {e}")

# 실행: 순수 이름만 입력하세요
delete_user_by_name("임형권")
