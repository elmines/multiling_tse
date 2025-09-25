from typing import List, Optional, Tuple
# 3rd Party
import torch
import numpy as np
from gensim.models import FastText
from transformers import PreTrainedTokenizerFast
from transformers.generation.utils import GenerateBeamEncoderDecoderOutput
from torch_scatter import segment_max_coo
# Local
from .constants import TARGET_DELIMITER, UNRELATED_TARGET

def make_target_embeddings(targets: List[str], fast_text: FastText) -> np.ndarray:
    unstacked = []
    for target in targets:
        if target == UNRELATED_TARGET:
            # Will never match with the unrelated target, because no vector X can have
            # a nonzero cosine similarity with the zero vector
            unstacked.append(np.zeros([fast_text.vector_size], dtype=np.float32))
        else:
            vec = fast_text.wv[target.lower()].copy()
            # Normalize magnitude to unity--makes computing explicit cosine similarity unnecessary
            # Just find the target vector that has the highest dot product with the predicted target
            vec = vec / np.linalg.norm(vec) 
            unstacked.append(vec)
    # (n_targets, embedding_size)
    stacked = np.stack(unstacked)
    return stacked

def detokenize(tokenizer: PreTrainedTokenizerFast, id_seq: List[int]) -> List[str]:
    full_string = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(id_seq, skip_special_tokens=True))
    target_names = full_string.split(TARGET_DELIMITER)
    pruned = []
    seen = set()
    for t in target_names:
        if t not in seen:
            pruned.append(t)
            seen.add(t)
    return pruned

def detokenize_generated_targets(generate_output: GenerateBeamEncoderDecoderOutput,
                                 tokenizer: PreTrainedTokenizerFast) -> Tuple[List[str], List[int]]:
    output_ids = generate_output.sequences
    sample_inds = []
    all_texts = []
    for i, id_seq in enumerate(output_ids.detach().cpu().tolist()):
        sample_targets = detokenize(tokenizer, id_seq)
        all_texts.extend(sample_targets)
        sample_inds.extend(i for _ in sample_targets)
    return all_texts, sample_inds

def map_targets(fast_text: FastText,
                target_embeddings: torch.Tensor,
                all_texts: List[str],
                sample_inds: torch.Tensor,
                related_threshold: float) -> Tuple[torch.Tensor, List[str]]:
    device = target_embeddings.device

    output_embeddings = fast_text.wv[all_texts]
    output_embeddings = torch.tensor(output_embeddings).to(device)
    output_embeddings = output_embeddings / torch.linalg.norm(output_embeddings, keepdim=True, dim=1)

    # For a given sample, we want to pick the fixed target that had the highest similarity score,
    # across any predicted target for that sample

    # 1. Calculate the similarity scores
    # (n_predicted_targets, n_fixed_targets)
    all_scores = output_embeddings @ target_embeddings.transpose(1, 0)
    # 2. Calculate the highest score for a given fixed target across the predicted targets for a sample
    # Different samples can have different numbers of targets predicted, so we have to use a sparse operation here
    # (n_samples, n_fixed_targets)
    sample_scores, arg_sample_scores = segment_max_coo(all_scores, sample_inds)
    # 3. Take the max across the fixed targets
    # (n_samples,) and (n_samples,)
    class_scores, arg_class_scores = torch.max(sample_scores, dim=-1)
    # 4. Pick the UNRELATED target if the class score doesn't meet the similarity threshold
    target_preds = torch.where(class_scores >= related_threshold, arg_class_scores, 0)

    # Get the string-based free-form targets as well
    all_text_inds = torch.gather(arg_sample_scores, 1, arg_class_scores.unsqueeze(1)).squeeze(1)
    freeform_preds = [all_texts[i] for i in all_text_inds]

    return target_preds, freeform_preds

