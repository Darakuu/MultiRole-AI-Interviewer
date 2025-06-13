import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Multi-Role Interview/Exam Agent

    This notebook implements a modular, cell-based interactive interview/exam simulator using the OpenAI API.

    ## Usage instructions:
    - Install Poetry,
    - Install the dependencies specified in the pyproject.toml file (WIP)
    - Set the `OPENAI_API_KEY` in your .env inside Marimo
    - Run the code!
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import os
    import re
    from openai import OpenAI
    return OpenAI, mo, re


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


@app.function
def ask_question(client, role, subject, is_exam, user_message):
    """
    Send a prompt to the model to ask exactly one question.
    Returns the generated question string.
    """
    mode = 'exam' if is_exam else 'interview'
    system_prompt = (
        f"You are a {role} conducting an oral {mode} on '{subject}'. "
        "Ask exactly one question, then wait for the candidate’s answer. "
        "Try to switch up the questions, do not always repeat the same question"
        "Try to surprise the candidate with questions he might not expect, while staying true to the subject."
        "If the candidate says 'quit' or 'exit', end politely."
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return resp.choices[0].message.content.strip()


@app.cell
def _(re):
    def grade_answer(client, question, answer):
        """
        If in exam mode, ask the model to grade the answer and return a float 0-30.
        """
        grade_prompt = (
            f"You are an unbiased examiner. Grade the following answer on a scale from 0 to 30 (just a number) "
            f"to the question: '{question}'. Answer: '{answer}'. "
            "The higher the grade, the better the answer. Grade the answer based on the accuracy and correctness of it, compared to the question. If the answer correctly answers the question, feel free to give 30 as a mark."
            "Respond with just the numeric grade."
        )
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": grade_prompt}],
        )
        try:
            return float(re.search(r"(\d+(?:\.\d+)?)", resp.choices[0].message.content).group(1))
        except Exception:
            return 0.0

    return (grade_answer,)


@app.cell
def _(OpenAI, grade_answer):
    def run_interview():
        """
        Main loop: initializes client, prompts subject/roles, cycles through roles asking questions,
        collects answers, optionally grades, and prints final results on quit.
        """
        client = OpenAI()
        subject, roles, is_exam = get_subject_and_roles()
        grades = []
        print(f"\n=== Starting {'EXAM' if is_exam else 'INTERVIEW'} on '{subject}' ===\n")

        idx = 0
        last_answer = None
        while True:
            role = roles[idx % len(roles)]
            user_msg = "START" if idx == 0 else last_answer
            question = ask_question(client, role, subject, is_exam, user_msg)
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
                grade = grade_answer(client, question, answer)
                grades.append(grade)
                print(f"Grade for this answer: {grade}/30\n")

            idx += 1
    return (run_interview,)


@app.cell
def _(run_interview):

    if __name__ == "__main__":
        run_interview()

    return


if __name__ == "__main__":
    app.run()
