import streamlit as st
import re
import os
import json
import time
from rich import print
from rich.console import Console
from loguru import logger
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.globals import set_verbose

set_verbose(False)

# Rich 콘솔 및 로거 초기화
console = Console()
logger.add("logs/file_{time}.log", rotation="500 MB")

# 프롬프트 템플릿 정의
PLAN_TEMPLATE = """
당신은 {course_name} {level} 과정의 전문 교육 플래너입니다.
학습 내용: {study_content}
다음 조건에 맞춰 학습 계획을 생성해주세요:
1. 학습 기간: {study_content}의 내용을 최대 20일로 나누어 계획을 수립하세요.
2. 각 일차별 계획은 다음과 같이 2줄로 요약하여 작성하세요:
- 학습 목표
- 핵심 학습 내용
3. {level} 수준에 맞게 난이도를 조절하여 설명해주세요.
출력 형식:
첫 줄: 총 학습 기간(일)
이후: 각 일차별 계획 (목표와 내용)
"""

DAILY_CONTENT_TEMPLATE = """
당신은 {course_name} {level} 과정의 전문 교육 도우미입니다.
학습 내용: {study_content}
일차: {day}
질문: {question}
각 난이도별 학습 내용 기준:
1. 1단계(유치원생): 놀이와 체험 중심, 시각적 자료 활용
2. 2단계(초등 저학년): 쉬운 용어, 그림과 예시 활용
3. 3단계(초등 고학년): 기본 개념 설명, 일상생활 연계
4. 4단계(중학생): 심화 개념, 실생활 응용
5. 5단계(고등학생): 전문적 개념, 체계적 구조
6. 6단계(대학생): 전공 수준, 실무 연계
7. 7단계(전문가): 최신 트렌드, 고급 이론
{level}수준에 맞게 다음 내용을 포함하여 설명해주세요:
1. 오늘의 학습 목표
2. 핵심 개념 설명
3. 실습 내용
4. 학습 확인 문제
"""

def init_llm():
    """LangChain 모델 초기화"""
    try:
        load_dotenv()
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Mistral API 키가 설정되지 않았습니다.")
        
        return ChatMistralAI(
            model="mistral-medium",  # 또는 mistral-small, mistral-large
            mistral_api_key=api_key,
            temperature=0.7
        )
    except Exception as e:
        logger.error(f"LLM 초기화 오류: {str(e)}")
        raise

def init_chains(llm):
    """프롬프트 체인 초기화"""
    plan_chain = PromptTemplate(
        template=PLAN_TEMPLATE,
        input_variables=["course_name", "level", "study_content"]
    ) | llm | StrOutputParser()
    
    daily_chain = PromptTemplate(
        template=DAILY_CONTENT_TEMPLATE,
        input_variables=["course_name", "level", "study_content", "day", "question"]
    ) | llm | StrOutputParser()
    
    return plan_chain, daily_chain

def init_session_data():
    return {
        'courses': {},
        'last_accessed_course': None
    }


def create_new_course(course_name, level, study_content, study_plan_response):
    duration_match = re.search(r'\d+', study_plan_response.split('\n')[0])
    duration = int(duration_match.group()) if duration_match else 20
    
    return {
        'course_name': course_name,
        'level': level,
        'study_content': study_content,
        'study_plan': study_plan_response,
        'created_at': time.strftime("%Y-%m-%d %H:%M:%S"),
        'last_access': time.strftime("%Y-%m-%d %H:%M:%S"),
        'duration': duration,
        'progress': {
            'current_day': 1,
            'completed_days': []
        },
        'daily_contents': {},
        'qa_history': []
    }


def save_session_data(session_id, data):
    try:
        os.makedirs('sessions', exist_ok=True)
        filename = f"sessions/session_{session_id}.json"
        serializable_data = {
            'courses': {},
            'last_accessed_course': data['last_accessed_course']
        }
        
        for course_id, course in data['courses'].items():
            serializable_data['courses'][course_id] = {
                'course_name': course['course_name'],
                'level': course['level'],
                'study_content': str(course['study_content']),
                'study_plan': str(course.get('study_plan', '')),
                'created_at': course['created_at'],
                'last_access': course['last_access'],
                'duration': course['duration'],
                'progress': course['progress'],
                'daily_contents': {k: str(v) for k, v in course['daily_contents'].items()},
                'qa_history': course['qa_history']
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"세션 저장 오류: {str(e)}")
        return False




def load_session_data(session_id):
    try:
        filename = f"sessions/session_{session_id}.json"
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 오류: {str(e)}")
                # 손상된 파일 백업
                backup_file = f"{filename}.backup"
                if os.path.exists(filename):
                    os.rename(filename, backup_file)
                return init_session_data()
        return init_session_data()
    except Exception as e:
        logger.error(f"세션 로드 오류: {str(e)}")
        return init_session_data()

def delete_session_data(session_id):
    """세션 데이터 삭제"""
    try:
        filename = f"sessions/session_{session_id}.json"
        if os.path.exists(filename):
            os.remove(filename)
            return True
        return False
    except Exception as e:
        logger.error(f"세션 삭제 오류: {str(e)}")
        return False

def main():
    st.set_page_config(layout="wide", page_title="LearnMate AI")
    
    # show_usage_guide 함수를 main 함수 시작 부분에 정의
    def show_usage_guide():
        st.markdown("## 🚀 LearnMate AI 사용 가이드")
        st.markdown("""
        1. 사이드바에 세션 ID를 입력하세요.
        2. 학습할 교육명, 학습 내용, 난이도를 설정하세요.
        3. '새로운 학습 계획 생성' 버튼을 클릭하여 AI가 학습 계획을 생성하도록 합니다.
        4. 생성된 학습 계획을 확인하고 일차별 학습을 진행하세요.
        5. 질문이 있다면 Q&A 섹션을 활용하세요.
        """)
    
    try:
        # 초기화
        llm = init_llm()
        plan_chain, daily_chain = init_chains(llm)
        
        if 'current_session' not in st.session_state:
            st.session_state.current_session = init_session_data()

            
        # UI 구현
        st.title('🎓 LearnMate AI')
        
        # 사이드바 구현
        with st.sidebar:
            session_id = st.text_input(
                "📝 세션 ID를 입력하세요",
                key="session_id",
                placeholder="예: user123"
            )
            
            if session_id:
                # 새로운 세션 ID 확인 및 초기화
                is_new_session = not os.path.exists(f"sessions/session_{session_id}.json")
                
                # 세션 상태에 따른 메시지 표시
                if is_new_session:
                    st.success(f"'{session_id}'님 반갑습니다!(신규)")
                    current_session = init_session_data()
                else:
                    st.info(f"'{session_id}'님 반갑습니다!(기존)")
                    current_session = load_session_data(session_id)
                
                st.session_state.current_session = current_session

                # 생성된 학습 계획 또는 마지막 학습 계획 표시 (새 위치)
                if 'new_plan' in st.session_state:
                    # 새로 생성된 학습 계획 표시
                    st.markdown("### 📅 새로운 학습 계획")
                    st.write(st.session_state.new_plan)
                    del st.session_state.new_plan  # 표시 후 삭제
                elif current_session.get('last_accessed_course'):
                    # 기존 세션의 마지막 학습 계획 표시
                    last_course = current_session['courses'][current_session['last_accessed_course']]
                    st.markdown(f"### 📋 {last_course['course_name']} 학습 계획")
                    st.write(last_course['study_plan'])  # study_content 대신 study_plan 사용

                # 학습 과정 설정 (구분선으로 분리)
                st.markdown("---")
                st.header("⚙️ 학습 과정 설정")

                
                # 학습 과정 입력 필드들
                course_name = st.text_input(
                    "학습할 교육명을 입력하세요!",
                    placeholder="예: 파이썬, 설득, 마케팅, 영어"
                )
                
                study_content = st.text_area(
                    "학습하고자 하는 내용을 자세히 입력하세요",
                    placeholder="예: 파이썬 기초 문법부터 시작해서 웹 개발까지 배우고 싶습니다."
                )
                
                level = st.selectbox(
                    "학습 LEVEL을 선택하세요",
                    ["1단계(유치원생)", "2단계(초등 저학년)", "3단계(초등 고학년)",
                    "4단계(중학생)", "5단계(고등학생)", "6단계(대학생)", "7단계(전문가)"]
                )
                
                if st.button('새로운 학습 계획 생성', use_container_width=True):
                    if not course_name:
                        st.warning("학습할 교육명을 입력하세요!")
                    elif course_name and study_content:
                        with st.spinner('학습 계획을 생성중입니다...'):
                            try:
                                # 새로운 학습 계획 생성
                                study_plan_response = plan_chain.invoke({
                                    "course_name": course_name,
                                    "level": level,
                                    "study_content": study_content
                                })
                                
                                # 새로운 과정 생성 및 저장
                                course_id = f"course_{len(current_session['courses']) + 1}"
                                current_session['courses'][course_id] = create_new_course(
                                    course_name,
                                    level,
                                    study_content,
                                    study_plan_response
                                )
                                
                                current_session['last_accessed_course'] = course_id
                                save_session_data(session_id, current_session)
                                
                                # 생성된 학습 계획을 세션 상태에 저장
                                st.session_state.new_plan = study_plan_response
                                st.rerun()  # 페이지 새로고침
                                
                            except Exception as e:
                                logger.error(f"학습 계획 생성 오류: {str(e)}")
                                st.error("학습 계획 생성 중 오류가 발생했습니다.")
                
                # 생성된 학습 계획 표시 (세션 상태 메시지 바로 아래)
                if 'new_plan' in st.session_state:
                    st.markdown("### 📅 새로운 학습 계획")
                    st.write(st.session_state.new_plan)

                
                # 기존 세션인 경우에만 삭제 버튼 표시
                if not is_new_session:
                    st.markdown("---")
                    if st.button("🗑️ 학습 DATA 삭제", type="secondary", use_container_width=True):
                        try:
                            if delete_session_data(session_id):
                                st.session_state.clear()
                                st.session_state.current_session = init_session_data()
                                st.session_state.session_id = ""
                                st.success("학습 데이터가 삭제되었습니다.")
                                st.rerun()
                        except Exception as e:
                            logger.error(f"데이터 삭제 오류: {str(e)}")
                            st.error("데이터 삭제 중 오류가 발생했습니다.")
        
        # 메인 화면
        if not session_id:
            show_usage_guide()
        elif is_new_session and not current_session.get('last_accessed_course'):
            show_usage_guide()
        elif session_id and current_session.get('last_accessed_course'):
            # 학습 내용 표시
            current_course = current_session['courses'][current_session['last_accessed_course']]
            main_col, qa_col = st.columns([7, 3])
            
            # 메인 컬럼 (학습 내용)
            with main_col:
                # level에서 단계만 추출
                level_number = current_course['level'].split('단계')[0]
                
                # study_content 문자열 처리
                study_content_text = current_course['study_content']
                truncated_content = (
                    study_content_text
                    if len(study_content_text) <= 10
                    else study_content_text[:9] + "~"
                )
                
                # 제목 표시
                st.markdown(
                    f"### 📚 {current_course['course_name']} "
                    f"({level_number}단계, {truncated_content})"
                )
                
                # 학습 일차 선택 슬라이더
                if is_new_session:
                    # 새로운 세션일 경우 1일차로 시작
                    selected_day = st.slider(
                        "학습 일차",
                        1,
                        current_course['duration'] or 20,
                        1
                    )
                else:
                    # 기존 세션일 경우 마지막 학습 일차로 설정
                    last_day = max(map(int, current_course['daily_contents'].keys())) if current_course.get('daily_contents') else 1
                    selected_day = st.slider(
                        "학습 일차",
                        1,
                        current_course['duration'] or 20,
                        last_day
                    )

                # 선택한 일차의 학습 내용
                st.markdown(f"### 📖 {selected_day}일차 학습 내용")
                if str(selected_day) in current_course['daily_contents']:
                    st.write(current_course['daily_contents'][str(selected_day)])
                else:
                    daily_content = daily_chain.invoke({
                        "course_name": current_course['course_name'],
                        "level": current_course['level'],
                        "study_content": current_course['study_content'],
                        "day": str(selected_day),
                        "question": ""
                    })
                    current_course['daily_contents'][str(selected_day)] = daily_content
                    save_session_data(session_id, current_session)
                    st.write(daily_content)

                # 마지막 학습 내용 표시 (기존 세션인 경우)
                if not is_new_session and current_course.get('daily_contents'):
                    last_day = max(map(int, current_course['daily_contents'].keys()))
                    st.markdown(f"### 📖 마지막 학습 내용 ({last_day}일차)")
                    st.write(current_course['daily_contents'][str(last_day)])

                # Q&A 섹션
                with qa_col:
                    st.markdown("### Q&A❓")
                    
                    # 채팅 메시지를 세션ID별로 저장
                    if 'chat_messages' not in st.session_state:
                        st.session_state.chat_messages = {}
                        
                    # 현재 세션ID의 메시지 초기화
                    if session_id not in st.session_state.chat_messages:
                        st.session_state.chat_messages[session_id] = []
                        
                    # 이전 질문-답변 기록 불러오기
                    if current_course.get('qa_history'):
                        # 최신순으로 정렬하여 메시지 추가
                        sorted_qa_history = sorted(
                            current_course['qa_history'],
                            key=lambda x: x['timestamp'],
                            reverse=True
                        )
                        st.session_state.chat_messages[session_id] = [{
                            "question": qa['question'],
                            "answer": qa['answer']
                        } for qa in sorted_qa_history]

                    # 질문 입력 영역
                    question = st.text_input(
                        label="질문 입력",
                        placeholder="질문을 입력하세요!",
                        key="question_input",
                        label_visibility="collapsed",
                        on_change=lambda: handle_question() if st.session_state.question_input else None
                    )

                    # 로딩 스피너를 위한 컨테이너
                    spinner_container = st.empty()

                    # 질문 처리 함수
                    def handle_question():
                        if st.session_state.question_input:
                            with spinner_container:
                                with st.spinner("답변을 생성중입니다..."):
                                    try:
                                        # 난이도에 따른 답변 생성을 위한 프롬프트
                                        level_prompt = f"""
                                        당신은 {current_course['level']} 수준의 학습자와 대화하고 있습니다.
                                        질문: {st.session_state.question_input}
                                        다음 기준에 맞춰 답변해주세요:
                                        1. 간결하고 명확하게 답변
                                        2. 해당 난이도에 맞는 용어와 설명 사용
                                        3. 최대 10문장 이하로 답변
                                        """
                                        
                                        # 답변 생성
                                        answer = llm.invoke(level_prompt)
                                        answer_content = answer.content
                                        
                                        # 새로운 Q&A를 리스트 시작 부분에 추가
                                        st.session_state.chat_messages[session_id].insert(0, {
                                            "question": st.session_state.question_input,
                                            "answer": answer_content
                                        })
                                        
                                        # Q&A 기록 저장
                                        current_course['qa_history'].insert(0, {
                                            'day': selected_day,
                                            'question': st.session_state.question_input,
                                            'answer': answer_content,
                                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                                        })
                                        save_session_data(session_id, current_session)
                                        
                                        # 입력 필드 초기화
                                        st.session_state.question_input = ""
                                        
                                    except Exception as e:
                                        logger.error(f"답변 생성 오류: {str(e)}")
                                        st.error("답변 생성 중 오류가 발생했습니다.")

                    # 현재 세션의 Q&A 메시지만 표시 (최신 순)
                    if session_id in st.session_state.chat_messages:
                        for message in st.session_state.chat_messages[session_id]:
                            st.markdown("---")
                            st.markdown(f"**Q: {message['question']}**")
                            st.markdown(f"A: {message['answer']}")

    except Exception as e:
        logger.error(f"애플리케이션 실행 오류: {str(e)}")
        st.error("애플리케이션 실행 중 오류가 발생했습니다.")
        return

if __name__ == "__main__":
    main()
