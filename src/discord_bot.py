import os
from dotenv import load_dotenv
import re

import discord
from discord.ext import commands
from stable_diffusion import generate_image, expand_image

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(intents=intents, command_prefix="/")


@bot.command()
async def generate(ctx, *args):
    message = get_cleaned_message(args)
    img = generate_image(message)
    await reply_and_cleanup_generated_img(ctx, message, img)


@bot.command()
async def prettify(ctx: discord.ext.commands.Context, *args):
    prompt = get_cleaned_message(args)
    sent_image = ctx.message.attachments[0]
    await sent_image.save(sent_image.filename)
    await expand_image_and_reply(sent_image, prompt, ctx)


async def expand_image_and_reply(sent_image, prompt, ctx):
    if sent_image and prompt != '':
        img = expand_image(str(sent_image.filename), prompt)
        await reply_and_cleanup_generated_img(ctx, prompt, img)
    else:
        await ctx.reply("No draft image or prompt to generate on was found! \n Please make sure you attached an image.")


def get_cleaned_message(args) -> str:
    message = ' '.join(args)
    return re.sub(r'\W ,-', '', message)


async def reply_and_cleanup_generated_img(ctx: discord.ext.commands.Context, prompt, img):
    path = os.path.normpath(prompt) + ".png"
    img.save(path)
    await ctx.reply(f"Your prompt: {prompt}", file=discord.File(path))
    os.remove(path)


@bot.event
async def on_message(message):
    await bot.process_commands(message)


if __name__ == "__main__":
    bot.run(TOKEN)
