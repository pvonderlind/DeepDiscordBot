# DeepDiscordBot

A small python script using ***discord.py*** and HuggingFace's implementation of the ***StableDiffusion*** text-to-image and image-to-image models.

Features:
* Adds a bot to your server which can be added to text channels
* The bot provides two commands `/generate` and `/prettify`

Using `/generate <prompt>` passes the prompt to the diffusion model. The generated image is then passed as a reply
to the prompt by the bot.

Using `/prettify` takes a prompt and an image to use as a starting vector for the diffusion model.  
The generated image is then passed as a reply to the prompt by the bot.

