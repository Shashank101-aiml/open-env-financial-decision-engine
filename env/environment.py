from env.models import Observation, Action, Reward
from tasks.loan_task import get_loan_task
from graders.loan_grader import grade_loan

class FinancialEnv:
    def __init__(self):
        self.current_task = None
        self.done = False

    def reset(self):
        self.current_task = get_loan_task()
        self.done = False
        return Observation(input_text=self.current_task["input"])

    def step(self, action: Action):
        score = grade_loan(action, self.current_task)

        reward = Reward(score=score)
        self.done = True

        return (
            Observation(input_text="Task completed"),
            reward,
            self.done,
            {}
        )

    def state(self):
        return self.current_task
