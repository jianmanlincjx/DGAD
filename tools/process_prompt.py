
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T
from PIL import Image

from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def remove_prompt_tags_and_save_to_file(text, file_path):
    # Remove the <prompt> tags from the beginning and end of the text
    cleaned_text = text.strip('<prompt>').strip('</prompt>')

    # Add "Natural light," at the beginning of the cleaned text
    cleaned_text = cleaned_text
    cleaned_text = cleaned_text

    # Write the cleaned text to a txt file
    with open(file_path, 'w') as file:
        file.write(cleaned_text)

def remove_prompt_tags_and_save_to_file_object(text, file_path):
    # Remove the <prompt> tags from the beginning and end of the text
    cleaned_text = text.strip('<prompt>').strip('</prompt>')

    # Add "Natural light," at the beginning of the cleaned text
    cleaned_text = cleaned_text
    cleaned_text = cleaned_text

    # Write the cleaned text to a txt file
    with open(file_path, 'w') as file:
        file.write(cleaned_text)


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

path = "/data/JM/code/BrushNet-main/pretrain_model/models--OpenGVLab--Mini-InternVL-Chat-4B-V1-5"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
generation_config = dict(
    num_beams=1,
    max_new_tokens=80,
    do_sample=False,
)



vid_path = f'/data/JM/code/BrushNet-main/validation_dataset/validation_input/object_cropped'
img_list = sorted(os.listdir(vid_path))

for img in tqdm(img_list):
    try:
        # single-round single-image conversation
        question = "请生成这幅图像的英文prompt用于stablediffusion生成,注意颜色的修饰，要求简短一点，80词内，输出的prompt需要以<prompt>开始并以</prompt>结束"
        img_path = f'{vid_path}/{img}'
        save_txt_path = img_path.replace('object_cropped', 'text').replace('.png', '.txt')
        pixel_values = load_image(img_path, max_num=6).to(torch.bfloat16).cuda()
        response = model.chat(tokenizer, pixel_values, question, generation_config)
        remove_prompt_tags_and_save_to_file(response, save_txt_path)
    except Exception as e:
        print(f"Error processing {img}: {e}")
        continue


# vid_path = f'/data1/JM/code/IP-Adapter-main/dataset/MSRA-10K/object'
# text_path = os.makedirs(vid_path.replace('object', 'text'), exist_ok=True)
# img_list = sorted(os.listdir(vid_path))

# for img in tqdm(img_list):
#     try:
#         # single-round single-image conversation
#         question = "Please generate an English prompt for Stable Diffusion based on this image, describing only the foreground. Do not mention the background. Focus on details like shape, color, and texture of the foreground. Keep the prompt under 80 words and ensure it starts with <prompt> and ends with </prompt>."
#         img_path = f'{vid_path}/{img}'
#         save_txt_path = img_path.replace('object', 'text').replace('.png', '.txt')
#         pixel_values = load_image(img_path, max_num=6).to(torch.bfloat16).cuda()
#         response = model.chat(tokenizer, pixel_values, question, generation_config)
#         remove_prompt_tags_and_save_to_file_object(response, save_txt_path)
#     except Exception as e:
#         print(f"Error processing {img}: {e}")
#         continue