# NewsImages at MediaEval 2025

## Task Description

Participants are given a collection of 8,500 news articles with images (the article text is in English, from [GDELT](https://www.gdeltproject.org).
Given a randomly selected article, the goal is to build a pipeline for (1) image retrieval or (2) image generation to provide a fitting image recommendation for a given news article.
The final evaluation event may only make use of subsets of the overall item pool that was shared with participating teams.
The relevant article IDs for the final evaluation will be communicated separately via email to all **registered groups**, together with the information on where to submit your results (see deadlines below).

Please see the official [MediaEval 2025 website](https://multimediaeval.github.io/editions/2025/tasks/newsimages) for the full task description and event registration.

## Data Description

The challenge data contains a CSV with the following data on news articles:

| Attribute | Description |
| - | -|
| article_id | ID of news article. |
| article_url | Original URL of the news article. |
| article_title | Title of the news article (may include lead). |
| article_tags | Automatically generated tags for the main article text/body. |
| image_id | ID of news image. |
| image_url | Original URL of the news image. |

Furthermore, a folder 'newsimages' containing a copy of all news images is included.
The name of each JPG file corresponds to the image ID associated with each news article.

## Expected Submission

Image retrieval and generation have two subtasks each, a small one (using pre-determined article IDs that will be communicated in advance) and a large one (using randomly selected article IDs).
The articles in both the small and large tasks are part of the dataset shared with participants.
For more details, please see the task [overview paper](https://github.com/Informfully/Challenges/blob/main/documents/newsimages_task_overview_paper.pdf).
You are requested to provide **at least one** approach for both the large and the small subtasks.
When creating a submission for the large subtask (e.g., "GEN_SD_LARGE"), please include/keep the image recommendation that this approach makes for IDs that are in the small subtask.

You must provide a ZIP file [group_name].zip that is structured as follows:

[group_name] / ["RET"|"GEN"] + _ + [approach_name] + _ + ["LARGE"|"SMALL"]/ [article_id] + _ + [group_name] + _ + [approach_name].png

Use the group name with which you have registered for the task.
For each submitted approach/run, please provide a **unique name**.
Use the prefix a to indicate the type ("RET" for retrieval approach, "GEN" for image generation) and a suffix for the subtask ("LARGE" covering all images and "SMALL" the pre-defined subset).
Your submission can include multiple approaches.

Each approach must include **precisely one** image recommendation for a given image ID.
It should cover and include **as many** many images as possible.
In case there is no image recommendation for a requested ID, this one image entry gets automatically assigned the lowest "image fit" score in the evaluation.

Example submission for the group 'UnstableOsmosis':

    UnstableOsmosis.zip
	|_ GEN_FLUX_SMALL
	|  |_ 37FC359AB91C0DC6D21D270AED0C87E3_UnstableOsmosis_FLUX.png
	|  |_ …
	|_ GEN_SD_LARGE
	|  |_ 37FC359AB91C0DC6D21D270AED0C87E3_UnstableOsmosis_SD.png
	|  |_ …
	|_ …

If you do not want to make a separate submission for the small challenge subtask, you can simply create a copy of the submissions for the large task, omitting any irrelevant IDs, by changing only the suffix (e.g., "GEN_SD_SMALL" is based on "GEN_SD_LARGE" but only includes recommendations for IDs of the small subtask).
Please note that approach names between the small and large subtasks cannot be shared unless it is precisely the same approach (i.e., the same image recommendations).
If there are small variations between the two, you need to give the approach a new name and clearly document this.

The image format must be PNG, with target dimensions of 460x260 pixels (landscape orientation).
This applies to both generated and retrieved images.
If you generate the images with tools like ComfyUI and you edit them afterwards (e.g., for cropping), make sure the workflow **remains** embedded.

Note on using original article images:
The final evaluation will make use of the dataset shared with participants.
Therefore, the original images for all articles are known.
It is not forbidden to leverage them for creating the retrieval/generation pipeline.
However, in our [previous study](https://ceur-ws.org/Vol-3658/paper8.pdf), we found that the original images often have a lower image fit than retrieved or generated ones.
And it is this image fit that will decide the winner in the final evaluation event.
Being able to retrieve the original image or generate something that looks similar is not relevant.

## Working Notes Paper

As part of the challenge submission, each team is required to write a separate **Working Notes Paper** that documents and outlines their approach.
Please look at the [online paper template](https://drive.google.com/drive/folders/1DNhxIeACfsmg6rrdgQZ22BbRtYE8ioYI) for more information.
We ask each group to include and refer to the following papers:

- [NewsImages in MediaEval 2025 – Comparing Image Retrieval and Generation for News Articles](https://github.com/Informfully/Challenges/blob/main/documents/newsimages_task_overview_paper.pdf), Heitz *et al.*, Working Notes Proceedings of the MediaEval 2025 Workshop, 2025.

  ```tex
  @inproceedings{heitz2025newsimages,
    title={NewsImages in MediaEval 2025 – Comparing Image Retrieval and Generation for News Articles},
    author={Lucien Heitz and Luca Rossetto and Benjamin Kille and Andreas Lommatzsch and Mehdi Elahi and Duc-Tien Dang-Nguyen},
    booktitle={Working Notes Proceedings of the MediaEval 2025 Workshop},
    year={2025}
  }
  ```

- [An Empirical Exploration of Perceived Similarity between News Article Texts and Images](https://ceur-ws.org/Vol-3658/paper8.pdf), Heitz *et al.*, Working Notes Proceedings of the MediaEval 2023 Workshop, 2024.

  ```tex
  @inproceedings{heitz2024empirical,
    title={An Empirical Exploration of Perceived Similarity between News Article Texts and Images},
    author={Lucien Heitz and Abraham Bernstein and Luca Rossetto},
    booktitle={Working Notes Proceedings of the MediaEval 2023 Workshop},
    year={2024}
  }
  ```

## Deadline Summary

* Runs due: September 10
* Working Notes Paper submission: October 8
* MediaEval workshop: October 25-26 (attendance required, in-person or online)

## Resources

* [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
* [WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
* [Yahoo-Flickr Creative Commons 100 Million (YFCC100M)](https://www.multimediacommons.org)
