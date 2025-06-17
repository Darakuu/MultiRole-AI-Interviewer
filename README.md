# Multi-Role Interview/Exam Agent

This notebook implements a modular, cell-based interactive interview/exam simulator using the OpenAI API.

## Usage instructions:
- Install Poetry,
- Install the dependencies specified in the pyproject.toml file
- Set the `OPENAI_API_KEY` in your .env inside Marimo
- Run the code!

## Exam vs Interview mode
> [!NOTE]
> - When choosing what type of subject you want to be examined on, writing '**exam**' anywhere in the input prompt will enable the grading feature. You will be graded on a scale from 0 to 30, and the final vote will be the average of all grades obtained during the exam.
> - Be careful, if your grades are too low, you will fail the exam!


> [!NOTE] 
> - If you do not write 'exam', you will instead be interviewed on your chosen subject by the roles you have added.
> - The interview will have a hidden score (being printed for debug purposes, comment it out if you want a more realistic experience!). You can either pass it, or fail. The LLM Agent judging the answer will penalize users attempting to game the system or using profanities in the answer. 
> [!IMPORTANT]
> To stop the program, write 'quit'