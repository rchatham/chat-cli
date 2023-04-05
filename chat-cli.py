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


def prepare_api_key():
    api_key = read_api_key()

    if not api_key:
        print("API key not found. Please enter your OpenAI API key.")
        api_key = input("API key: ").strip()
        write_api_key(api_key)

    openai.api_key = api_key


def get_user_input():
    try:
        return click.prompt(click.style("You", fg='green', bold=True))
    except click.exceptions.Abort:
        raise KeyboardInterrupt


def create_chat_params(model, messages, options):
    params = {
        "model": model,
        "messages": messages,
    }

    for key, value in options.items():
        if value is not None:
            params[key] = value

    return params



def print_assistant_response(response):
    click.echo(click.style("Assistant:", fg='yellow', bold=True))

    assistant_response = ""
    for chunk in response:
        chunk_text = chunk["choices"][0].get("delta", {}).get("content", "")
        if chunk_text:
            assistant_response += chunk_text
            click.echo(click.style(f"{chunk_text}", fg='yellow'), nl=False)

    click.echo("")  # Add a newline at the end
    return assistant_response.strip()


@click.command()
@click.option("--model", default="gpt-4", help="The model to use.")
@click.option("--system_message", default="You are a cli chat bot using OpenAI's API.", help="A custom system message.")
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
def start_chat(model, system_message, temperature, top_p, n, stream, stop, max_tokens, presence_penalty,
               frequency_penalty, logit_bias, user):
    click.echo(click.style(f"System: {system_message}", fg='yellow', bold=True))

    messages = [{"role": "system", "content": system_message}]

    options = {
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "stream": stream,
        "stop": stop,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": json.loads(logit_bias) if logit_bias else None,
        "user": user
    }

    while True:
        try:
            user_input = get_user_input()
            messages.append({"role": "user", "content": user_input})

            params = create_chat_params(model, messages, options)

            response = openai.ChatCompletion.create(**params)

            assistant_response = print_assistant_response(response)
            messages.append({"role": "assistant", "content": assistant_response})

        except (KeyboardInterrupt, EOFError):
            click.echo("\nExiting the chat session.")
            break


if __name__ == "__main__":
    prepare_api_key()
    start_chat()