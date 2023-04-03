import click
import openai
import json
import configparser
import os

config_file = 'config.ini'


def read_api_key():
    config = configparser.ConfigParser()
    config.read(config_file)
    try:
        return config.get('openai', 'api_key')
    except (configparser.NoSectionError, configparser.NoOptionError):
        return None


def write_api_key(api_key):
    config = configparser.ConfigParser()
    config.read(config_file)
    if not config.has_section('openai'):
        config.add_section('openai')
    config.set('openai', 'api_key', api_key)

    with open(config_file, 'w') as f:
        config.write(f)


OPENAI_API_KEY = read_api_key()

if not OPENAI_API_KEY:
    print("API key not found. Please enter your OpenAI API key.")
    OPENAI_API_KEY = input("API key: ").strip()
    write_api_key(OPENAI_API_KEY)

openai.api_key = OPENAI_API_KEY

@click.command()
@click.option("--model", default="gpt-4", help="The model to use.")
@click.option("--custom_message", default="You are a cli chat bot using OpenAI's API.", help="A custom system message.")
@click.option("--temperature", default=None, type=float, help="The sampling temperature.")
@click.option("--top_p", default=None, type=float, help="The nucleus sampling value.")
@click.option("--n", default=None, type=int, help="The number of chat completion choices.")
@click.option("--stream", default=True, type=bool, help="Enable partial message deltas streaming.")
@click.option("--stop", default=None, type=str, help="The stop sequence(s) for the API.")
@click.option("--max_tokens", default=None, type=int, help="The maximum number of tokens to generate.")
@click.option("--presence_penalty", default=None, type=float, help="The presence penalty.")
@click.option("--frequency_penalty", default=None, type=float, help="The frequency penalty.")
@click.option("--logit_bias", default=None, type=str, help="The logit bias as a JSON string.")
@click.option("--user", default=None, type=str, help="A unique identifier for the end-user.")
def start_chat(model, custom_message, temperature, top_p, n, stream, stop, max_tokens, presence_penalty,
               frequency_penalty, logit_bias, user):
    click.echo(click.style(custom_message, fg='yellow', bold=True))

    messages = []

    while True:
        try:
            user_input = click.prompt(click.style("You", fg='green', bold=True))
            messages.append({"role": "user", "content": user_input})

            params = {
                "model": model,
                "messages": messages,
            }

            if temperature: params["temperature"] = temperature
            if top_p: params["top_p"] = top_p
            if n: params["n"] = n
            if stream: params["stream"] = stream
            if stop: params["stop"] = stop
            if max_tokens: params["max_tokens"] = max_tokens
            if presence_penalty: params["presence_penalty"] = presence_penalty
            if frequency_penalty: params["frequency_penalty"] = frequency_penalty
            if logit_bias: params["logit_bias"] = json.loads(logit_bias)
            if user: params["user"] = user

            response = openai.ChatCompletion.create(**params)
            click.echo(click.style("Assistant:", fg='yellow', bold=True))

            # prev_char = None
            for chunk in response:
                chunk_text = chunk["choices"][0].get("delta", {}).get("content", "")
                if chunk_text:
                    click.echo(click.style(f"{chunk_text}", fg='yellow'), nl=False)

            click.echo("")  # Add a newline at the end

        except (KeyboardInterrupt, EOFError):
            click.echo("\nExiting the chat session.")
            break


if __name__ == "__main__":
    start_chat()
