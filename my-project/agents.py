import ast
import json
import operator
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from lib.messages import UserMessage, SystemMessage, ToolMessage
from lib.tooling import tool, Tool
from lib.llm import LLM

load_dotenv()

_CALC_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    # Whitelist-only AST walk so expressions can't call functions or access names/builtins.
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _CALC_OPERATORS:
        return _CALC_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_OPERATORS:
        return _CALC_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool
def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression, e.g. '2 + 3 * 4'."""
    tree = ast.parse(expression, mode="eval")
    return _safe_eval(tree.body)


# Placeholder sample data; swap in a real games data source (file, DB, API) when available.
_SAMPLE_GAMES = [
    {"name": "Lakers vs Celtics", "date": "2026-09-01", "score": 132},
    {"name": "Warriors vs Suns", "date": "2026-09-05", "score": 128},
    {"name": "Bulls vs Heat", "date": "2026-09-08", "score": 97},
    {"name": "Nets vs Knicks", "date": "2026-09-10", "score": 145},
    {"name": "Mavericks vs Nuggets", "date": "2026-09-12", "score": 110},
]


@tool
def get_games(num_games: int = 1, top: bool = True) -> str:
    """Return the `num_games` highest- (or lowest-, if top=False) scoring games."""
    ranked = sorted(_SAMPLE_GAMES, key=lambda g: g["score"], reverse=top)
    selected = ranked[:num_games]
    return "\n".join(f"{g['name']} ({g['date']}): {g['score']} pts" for g in selected)


class Agent:
    def __init__(
        self,
        role: str,
        instructions: str = "Be helpful, concise, and accurate.",
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        tools: Optional[List[Tool]] = None,
    ):
        self.role = role
        self.instructions = instructions
        self.tools = tools or []
        self.llm = LLM(model=model, temperature=temperature)

    def invoke(self, user_message: str) -> str:
        messages: List[Any] = [
            SystemMessage(content=f"You are {self.role}. {self.instructions}"),
            UserMessage(content=user_message),
        ]

        response = self.llm.invoke(messages, tools=self.tools)
        while response.tool_calls:
            messages.append(response)
            for call in response.tool_calls:
                result = self._run_tool(call)
                messages.append(ToolMessage(content=str(result), tool_call_id=call.get("id", "")))
            response = self.llm.invoke(messages, tools=self.tools)

        return response.content

    def _run_tool(self, call: Dict[str, Any]) -> Any:
        function = call.get("function", {})
        tool_name = function.get("name")
        tool_args = json.loads(function.get("arguments") or "{}")
        matched = next((t for t in self.tools if t.name == tool_name), None)
        if matched is None:
            return f"Tool '{tool_name}' not found"
        return matched.run(**tool_args)


if __name__ == "__main__":
    agent = Agent(role="Coding Assistant")
    response = agent.invoke("What is Python? Be concise")
    print(response)

    math_agent = Agent(role="Math Assistant", tools=[calculate])
    response = math_agent.invoke("What is 23 * 45?")
    print(response)

    data_analyst_agent = Agent(role="Game Stats Assistant", tools=[get_games])
    response = data_analyst_agent.invoke("What's the best game in the dataset?")
    print(response)
