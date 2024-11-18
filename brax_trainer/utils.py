from PIL import ImageDraw
from moviepy.editor import ImageSequenceClip


def init_wandb(args_dict, run_name, id=None, resume=True, disable=False):
    import wandb

    if disable is True:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            id=id or wandb.util.generate_id(),
            project=args_dict["wandb_project"],
            group=args_dict["wandb_group"],
            allow_val_change=True,
            save_code=True,
            resume=resume,
            config=args_dict,
            name=run_name,
        )

    return wandb


# Functions for video generation
def add_text_to_image(image, text, position, color=(255, 255, 255)):
    """Add text to an image using PIL."""
    draw = ImageDraw.Draw(image)
    draw.text(position, text, fill=color)
    return image


def create_video(frames, output_path, fps=30):
    clip = ImageSequenceClip(list(frames), fps=fps)
    clip.write_videofile(output_path, codec="libx264")
