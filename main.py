from typing import List, Dict, Any, Optional
import os
import json

from dotenv import load_dotenv
from langsmith import Client
from langsmith.wrappers import wrap_openai
from langchain_openai import ChatOpenAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI

from prompts import ASSISTANT_DESCRIPTION_PROMPT

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")


# Data for the recipe database
def load_recipes() -> List[Dict[str, Any]]:
    recipies = None
    with open("./recipes.json", "r") as f:
        recipes = json.load(f)
    return recipes


RECIPIES = load_recipes()


# llm = ChatOpenAI(model=MODEL_NAME, temperature=0.7)
# llm = ChatMistralAI(
#     model="mistral-medium",
#     api_key=os.getenv("MISTRAL_API_KEY"),
#     temperature=0.7,
# )

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
)

# Set of tools available to the llm for recipies


@tool
def search_recipes_by_ingredients(ingredients: List[str]) -> List[Dict[str, Any]]:
    """
    Search for recipes that include the specified ingredients.
    Args:
        ingredients: A list of ingredient names
    Returns:
        A list of recipes that include at least one of the specified ingredients
    """
    results = []
    ingredients_lower = [ing.lower() for ing in ingredients]

    for recipe in RECIPIES:
        if any(
            ing in [r_ing.lower() for r_ing in recipe["ingredients"]]
            for ing in ingredients_lower
        ):
            results.append(recipe)

    return results


@tool
def search_recipes_by_cuisine(cuisine: str) -> List[Dict[str, Any]]:
    """
    Search for recipes of a specific cuisine.
    Args:
        cuisine: The cuisine type to search for
    Returns:
        A list of recipes matching the specified cuisine
    """
    results = []
    cuisine_lower = cuisine.lower()

    for recipe in RECIPIES:
        if recipe["cuisine"].lower() == cuisine_lower:
            results.append(recipe)

    return results


@tool
def search_recipes_by_dietary(dietary_preference: str) -> List[Dict[str, Any]]:
    """
    Search for recipes matching a specific dietary preference.
    Args:
        dietary_preference: The dietary preference to filter by (e.g., "vegetarian", "gluten-free", etc.)
    Returns:
        A list of recipes matching the specified dietary preference
    """
    results = []
    preference_lower = dietary_preference.lower()

    for recipe in RECIPIES:
        if preference_lower in [pref.lower() for pref in recipe["dietary"]]:
            results.append(recipe)

    return results


@tool
def get_recipe_details(recipe_id: int) -> Dict[str, Any]:
    """
    Get detailed information about a specific recipe.
    Args:
        recipe_id: The ID of the recipe to retrieve
    Returns:
        The recipe details if found, or an error message
    """
    for recipe in RECIPIES:
        if recipe["id"] == recipe_id:
            return recipe

    return {"error": f"Recipe with id {recipe_id} not found"}


@tool
def adapt_recipe(
    recipe_id: int,
    to_add: Optional[List[str]] = None,
    to_remove: Optional[List[str]] = None,
    dietary_restrictions: Optional[List[str]] = None,
) -> str:
    """
    Adapt a recipe by adding or removing ingredients or accommodating dietary restrictions.
    Args:
        recipe_id: The ID of the recipe to adapt
        to_add: Optional list of ingredients to add
        to_remove: Optional list of ingredients to remove
        dietary_restrictions: Optional list of dietary restrictions to accommodate
    Returns:
        A description of the adapted recipe
    """
    recipe = None
    for r in RECIPIES:
        if r["id"] == recipe_id:
            recipe = r
            break

    if not recipe:
        return f"Recipe with Id {recipe_id} not found"

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are a professional chef who specializes in adapting recipes to meet specific requirements."
            ),
            HumanMessage(
                content=f"""
        I need to adapt this recipe: {recipe['name']}
        
        Original ingredients: {', '.join(recipe['ingredients'])}
        
        Changes to make:
        {f"Add these ingredients: {', '.join(to_add)}" if to_add else ""}
        {f"Remove these ingredients: {', '.join(to_remove)}" if to_remove else ""}
        {f"Accommodate these dietary restrictions: {', '.join(dietary_restrictions)}" if dietary_restrictions else ""}
        
        Please provide a modified version of the recipe with adjusted ingredients, steps, and cooking instructions.
        """
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({})

    return result


# Setup the agent with tools
tools = [
    search_recipes_by_ingredients,
    search_recipes_by_cuisine,
    search_recipes_by_dietary,
    get_recipe_details,
    adapt_recipe,
]


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ASSISTANT_DESCRIPTION_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)


# def formatted(content: str) -> str:
#     words = content.split()
#     count = 0  # 80 chars per line
#     for word in words:
#         if count + len(word) + 1 > 80:
#             print("\n" + word, end=" ")
#             count = len(word) + 1
#         else:
#             print(word, end=" ")
#             count += len(word) + 1
#     print()


def main():
    history = []

    print("\n===== 🍽️ Recipe Assistant =====")
    print(
        "Welcome to Recipe Assistant! I'll help you find and adapt recipes based on your preferences."
    )
    print("Type 'exit' to end the conversation.\n")

    while True:
        query = input("\nYou: ")
        print()

        if query.strip().lower() == "exit":
            print("\nThank you for using Recipe Assistant! Goodbye!")
            break

        history.append({"role": "user", "content": query})

        print("Assistant: Thinking", end="\r")

        try:
            # Call the agent executor with the user query
            # response = agent_executor.invoke({"input": query})
            response = agent_executor.invoke(
                {
                    "input": query,
                    "agent_scratchpad": [],  # Required placeholder
                    "chat_history": [
                        (
                            HumanMessage(content=h["content"])
                            if h["role"] == "user"
                            else SystemMessage(content=h["content"])
                        )
                        for h in history
                    ],
                }
            )
            content = response["output"]

            print(f"Assistant: {content}")
            history.append({"role": "assistant", "content": content})
        except Exception as e:
            print(f"Assistant: Sorry, I encountered an error: {str(e)}")
            print()


if __name__ == "__main__":
    main()
