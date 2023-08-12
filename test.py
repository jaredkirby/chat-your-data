from langchain_decorators import llm_prompt

from secret import openai_api_key


@llm_prompt
def write_me_short_post(
    topic: str, platform: str = "twitter", audience: str = "developers"
):
    """
    Write me a short header for my post about {topic} for {platform} platform.
    It should be for {audience} audience.
    (Max 15 words)
    """
    pass


write_me_short_post(topic="Releasing a new App that can do real magic!")
