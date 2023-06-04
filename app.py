# app.py
import model
import func

model = model.model

text = "김철기 교수님 번호좀 알려줘"
print(func.find_answer(text, func.intent(model,text))) # 질문에 대한 답변 문자열