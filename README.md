# NewsImages at MediaEval 2025

## Task Description
Participants are given a collection of 8,500 news articles with images (the article text is in English, from [GDELT](https://www.gdeltproject.org)).
Given a randomly selected article, the goal is to build a pipeline for a) image retrieval or b) image generation to provide a fitting image recommendation for a given news article. Depending on the number of participants, the final evaluation event may only make use of  a subset of items.
Please see the official [MediaEval 2025 website](https://multimediaeval.github.io/editions/2025/tasks/newsimages) for the full task description.

## Data Description
The challenge data contains a CSV with the following data on news articles:

| Attribute | Description |
| - | - |
| article_id | ID of news article. |
| article_url | Original URL of the news article. |
| article_title | Title of the news article (may include lead). |
| article_tags | Automatically generated tags for main article text. |
| image_id | ID of news image. |
| image_url | Original URL of the news image. |

Furthermore, a folder 'newsimages' containing a copy of all news images is included.
The name of each JPG files corresponds to the image ID associated with each news article.

## Expected Submission
You must provide a ZIP file [group_name].zip that is organized as follows:

[group_name] / [approach_name] / [image_id] + _ + [group_name] + _ + [approach_name].png

Your shubmission should include as **many** image recommendations for the list of requested article IDs as possible.
The article pool relevant for this final evaluation will be communicated separately (see deadlines below).

Example submission for the group 'UnstableOsmosis':

    UnstableOsmosis.zip
	|_ FLUX
	|  |_ 37FC359AB91C0DC6D21D270AED0C87E3_UnstableOsmosis_FLUX.png
	|  |_ …
	|_ …

The image format must be PNG, with target dimensions of 460x260 pixels (i.e., landscape orientation).
This applies to both generated and retrieved images. If you generate the images with tools like ComfyUI and you edit them afterwards (e.g., for cropping), make sure the workflow **remains** embedded.

## Deadline Summary
* Data release: June 2
* Item pool announcement: August 4
* Runs due: September 10
* Paper submission: October 8
* MediaEval workshop: October 25-26 (attendance required, in-person or online)


## Resources
* [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
* [WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
* [Yahoo-Flickr Creative Commons 100 Million (YFCC100M)](https://www.multimediacommons.org)
