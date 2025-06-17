import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Multi-Role Interview/Exam Agent

    This notebook implements a modular, cell-based interactive interview/exam simulator using the OpenAI API.

    ## Usage instructions:
    - Install Poetry,
    - Install the dependencies specified in the pyproject.toml file
    - Set the `OPENAI_API_KEY` in your .env inside Marimo
    - Run the code!
    - **NOTE**: when choosing what type of subject you want to be examined on, writing '**exam**' anywhere in the input prompt will enable the grading feature. You will be graded on a scale from 0 to 30, and the final vote will be the average of all grades obtained during the exam.
        - Be careful, if your grades are too low, you will fail the exam!
    - **NOTE**: If you do not write 'exam', you will instead be interviewed on your chosen subject by the roles you have added.
        - The interview will have a hidden score (being printed for debug purposes, comment it out if you want a more realistic experience!). You can either pass it, or fail. The LLM Agent judging the answer will penalize users attempting to game the system or using profanities in the answer. 
    - To stop the program, write 'quit'
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import os
    import re

    from openai import OpenAI
    from langchain_community.chat_models import ChatOpenAI

    from langchain.chains import LLMChain
    from langchain.memory import ConversationBufferMemory
    from langchain.agents import Tool
    from langchain.prompts import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    from langchain.schema import SystemMessage
    return (
        ChatOpenAI,
        ChatPromptTemplate,
        ConversationBufferMemory,
        HumanMessagePromptTemplate,
        LLMChain,
        OpenAI,
        SystemMessage,
        SystemMessagePromptTemplate,
        Tool,
        mo,
        re,
    )


@app.cell
def _(OpenAI):
    # single global OpenAI client.
    client = OpenAI()
    return


@app.function
def get_subject_and_roles():
    """
    Prompt the user for subject and roles, return subject, roles list, and is_exam flag.
    """
    subject = input("Enter the exam/interview subject: ").strip()
    roles_input = input(
        "Enter interviewer roles (comma-separated, or blank for defaults HR, Technical Expert, Professor, Examiner): "
    ).strip()
    if roles_input:
        roles = [r.strip() for r in roles_input.split(",") if r.strip()]
    else:
        roles = ["HR", "Technical Expert", "Professor", "Examiner"]
    is_exam = any("exam" in subject.lower() or "exam" in r.lower() for r in roles)
    return subject, roles, is_exam


@app.cell
def _(
    ChatOpenAI,
    ChatPromptTemplate,
    ConversationBufferMemory,
    HumanMessagePromptTemplate,
    LLMChain,
    SystemMessagePromptTemplate,
    Tool,
):
    def make_role_agent(role: str, subject: str, is_exam: bool) -> Tool:
            mode = "exam" if is_exam else "interview"

            # pick the right system prompt depending on exam vs interview
            if is_exam:
                system_template = (
                    f"You are a **{role}** conducting an oral **{mode}** on “{subject}.”\n"
                    "Ask exactly **one** question, then stop and wait for the candidate’s reply.\n"
                    "Vary your questions and stay true to your role."
                )
            else:
                system_template = (
                    f"You are a **{role}** conducting an oral **{mode}** on “{subject}.”\n"
                    "When given the candidate’s previous answer, first provide a brief comment on their response, "
                    "then ask exactly **one** follow-up question that builds on what they said. "
                    "If this is the very first turn (input = START), just ask the opening question.\n"
                )

            system_msg = SystemMessagePromptTemplate.from_template(system_template)
            human_msg  = HumanMessagePromptTemplate.from_template("{user_input}")
            chat_prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])

            chain = LLMChain(
                llm=ChatOpenAI(model_name="gpt-4o", temperature=0.7),
                prompt=chat_prompt,
                memory=ConversationBufferMemory(memory_key="chat_history"),
            )

            # keep track of every question/comment combo to avoid exact repeats
            seen: set[str] = set()

            def ask_unique(user_input: str) -> str:
                for _ in range(6):
                    out = chain.run(user_input=user_input).strip()
                    if out not in seen:
                        seen.add(out)
                        return out
                return out

            return Tool(
                name=role,
                func=ask_unique,
                description=f"Comments & asks next question in the style of a {role}.",
            )
    return (make_role_agent,)


@app.cell
def _(ChatOpenAI, SystemMessage, re):
    def evaluate_hidden_score(question: str, answer: str) -> int:
        """
        Returns +1, 0, or -2 based on the candidate’s answer.
        Always returns an integer.
        """
        prompt = (
            "You are a fair scorer. The candidate just answered:\n\n"
            f"Q: {question}\n"
            f"A: {answer}\n\n"
            "If the answer is on point and professional, return +1.  "
            "If it’s somewhat off or contains minor issues, return 0.  "
            "If it’s very poor, off-topic, profane, or clearly gaming the system, return -2.  "
            "Output just the number."
        )
        chat = ChatOpenAI(model_name="gpt-4o", temperature=0)
        ai_msg = chat.invoke([SystemMessage(content=prompt)])
        text = ai_msg.content or ""
        m = re.search(r"(-?\d+)", text)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return 0
        # fallback if no match
        return 0
    return (evaluate_hidden_score,)


@app.cell
def _(ChatOpenAI, SystemMessage, re):
    def grade_answer(question: str, answer: str) -> float:
        """
        If in EXAM mode, ask the model to grade the answer and return a float 0–30.
        """
        grade_prompt = (
            f"You are an unbiased examiner. Grade the following answer on a scale from 0 to 30 (just a number) "
            f"to the question: '{question}'. Answer: '{answer}'. "
            "The better the answer, the higher the grade. Grade based on accuracy and correctness. "
            "Respond with just the numeric grade."
        )

        chat = ChatOpenAI(model_name="gpt-4o", temperature=0)

        ai_msg = chat.invoke([SystemMessage(content=grade_prompt)])

        text = ai_msg.content

        # Try to find a numeric grade in the response text
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if match:
            # Convert the captured substring to a float and return it
            return float(match.group(1))

        # If no number was found, default to zero
        return 0.0

    return (grade_answer,)


@app.cell
def _(evaluate_hidden_score, grade_answer, make_role_agent):
    def run_interview():
        subject, roles, is_exam = get_subject_and_roles()
        # build one agent per role
        agents = [make_role_agent(r, subject, is_exam) for r in roles]

        print(f"\n=== Starting {'EXAM' if is_exam else 'INTERVIEW'} on '{subject}' ===\n")

        grades = []
        consec_fails = 0 # EXAM: consecutive grades <= 10 tracker
        hidden_score = 0 # INTERVIEW: hidden-score tracker
        threshold = -5   # INTERVIEW: fail if we dip below this value
        positive_threshold = 5 # INTERVIEW: pass the interview if we go over this value
        idx = 0
        last_answer = None

        while True:
            role = roles[idx % len(roles)]
            # pull the matching Tool so we keep the order        
            tool = next(t for t in agents if t.name == role)

            user_msg = "START" if idx == 0 else last_answer

            # each call to tool.func() runs its own LLMChain + memory
            question = tool.func(user_msg)
            print(f"{role} ➜ {question}\n")

            answer = input("Your answer (or 'quit' to exit): ").strip()
            if answer.lower() in ("quit", "exit"):
                if is_exam and grades:
                    avg = sum(grades) / len(grades)
                    print(f"\n=== Exam ended. Your final grade: {avg:.1f}/30 ===")
                else:
                    print("\n=== Interview ended. Thank you! ===")
                break

            last_answer = answer

            if is_exam:
                g = grade_answer(question, answer)
                grades.append(g)
                print(f"Grade for this answer: {g}/30\n")

                # update consecutive failures
                if g < 10:
                    consec_fails += 1
                else:
                    consec_fails = 0

                # fail after 3 low scores in a row
                if consec_fails >= 3:
                    # pick one of the roles to deliver the news
                    fail_role = "Examiner" if "Examiner" in roles else roles[-1]
                    print(f"{fail_role} ➜ I’m sorry, but after three consecutive scores below 10, you have failed the exam.\n")
                    print("\n=== Exam failed. Better luck next time! ===")
                    break
            else:
                # non-exam: evaluate hidden score
                delta = evaluate_hidden_score(question, answer)
                hidden_score += delta
                judge = "Interview Judge"
            
                # DEBUG SCORE:
                print(f"Hidden Score is now: [ {hidden_score} ], changed by [{delta}] ")
            
                if hidden_score < threshold:
                    print(f"{judge} ➜ Unfortunately, based on the conversation so far, we must end the interview.\n")
                    print("\n=== Interview failed. Thank you for your time! ===")
                    break

                # succeed if high enough
                if hidden_score >= positive_threshold:
                    print(f"{judge} ➜ Congratulations! Based on your responses, you’ve demonstrated strong communication and fit.\n")
                    print("\n=== Interview passed! Well done! ===")
                    break

            idx += 1
    return (run_interview,)


@app.cell
def _(run_interview):

    if __name__ == "__main__":
        run_interview()

    return


if __name__ == "__main__":
    app.run()
