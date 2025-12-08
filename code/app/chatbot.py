from rag_agent import ask_agent, AGENT_ADMISSION

question = "Comment intégrer l’ESILV après un BUT informatique ?"

answer = ask_agent(question, AGENT_ADMISSION)

print(answer)