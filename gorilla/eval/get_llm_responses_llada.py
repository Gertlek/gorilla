
import argparse
import os
import sys
import json
import time
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from transformers import AutoTokenizer, AutoModel

# Add retrievers directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrievers import BM25Retriever, GPTRetriever
from retrievers.build_json_index import JSONLReader

# System prompt for API writing (matching retriever version)
SYSTEM_PROMPT = "You are a helpful API writer who can write APIs based on requirements."


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf.
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def encode_question(question, api_name, retrieved_docs=None):
    """Encode multiple prompt instructions into a single string.
    
    Returns:
        tuple: (base_prompt, full_prompt_with_docs) for logging purposes
    """
    
    if api_name == "torchhub":
        domains = "1. $DOMAIN is inferred from the task description and should include one of {Classification, Semantic Segmentation, Object Detection, Audio Separation, Video Classification, Text-to-Speech}."
    elif api_name == "huggingface":
        domains = "1. $DOMAIN should include one of {Multimodal Feature Extraction, Multimodal Text-to-Image, Multimodal Image-to-Text, Multimodal Text-to-Video, \
        Multimodal Visual Question Answering, Multimodal Document Question Answer, Multimodal Graph Machine Learning, Computer Vision Depth Estimation,\
        Computer Vision Image Classification, Computer Vision Object Detection, Computer Vision Image Segmentation, Computer Vision Image-to-Image, \
        Computer Vision Unconditional Image Generation, Computer Vision Video Classification, Computer Vision Zero-Shor Image Classification, \
        Natural Language Processing Text Classification, Natural Language Processing Token Classification, Natural Language Processing Table Question Answering, \
        Natural Language Processing Question Answering, Natural Language Processing Zero-Shot Classification, Natural Language Processing Translation, \
        Natural Language Processing Summarization, Natural Language Processing Conversational, Natural Language Processing Text Generation, Natural Language Processing Fill-Mask,\
        Natural Language Processing Text2Text Generation, Natural Language Processing Sentence Similarity, Audio Text-to-Speech, Audio Automatic Speech Recognition, \
        Audio Audio-to-Audio, Audio Audio Classification, Audio Voice Activity Detection, Tabular Tabular Classification, Tabular Tabular Regression, \
        Reinforcement Learning Reinforcement Learning, Reinforcement Learning Robotics }"
    elif api_name == "tensorhub":
        domains = "1. $DOMAIN is inferred from the task description and should include one of {text-sequence-alignment, text-embedding, text-language-model, text-preprocessing, text-classification, text-generation, text-question-answering, text-retrieval-question-answering, text-segmentation, text-to-mel, image-classification, image-feature-vector, image-object-detection, image-segmentation, image-generator, image-pose-detection, image-rnn-agent, image-augmentation, image-classifier, image-style-transfer, image-aesthetic-quality, image-depth-estimation, image-super-resolution, image-deblurring, image-extrapolation, image-text-recognition, image-dehazing, image-deraining, image-enhancemenmt, image-classification-logits, image-frame-interpolation, image-text-detection, image-denoising, image-others, video-classification, video-feature-extraction, video-generation, video-audio-text, video-text, audio-embedding, audio-event-classification, audio-command-detection, audio-paralinguists-classification, audio-speech-to-text, audio-speech-synthesis, audio-synthesis, audio-pitch-extraction}"
    else:
        print("Error: API name is not supported.")
        domains = ""

    base_prompt = question + "\nWrite a python program in 1 to 2 lines to call API in " + api_name + ".\n\nThe answer should follow the format: <<<domain>>> $DOMAIN, <<<api_call>>>: $API_CALL, <<<api_provider>>>: $API_PROVIDER, <<<explanation>>>: $EXPLANATION, <<<code>>>: $CODE}. Here are the requirements:\n" + domains + "\n2. The $API_CALL should have only 1 line of code that calls api.\n3. The $API_PROVIDER should be the programming framework used.\n4. $EXPLANATION should be a step-by-step explanation.\n5. The $CODE is the python code.\n6. Do not repeat the format in your answer."
    
    # Add retrieved API documentation if available (matching retriever version wording)
    full_prompt = base_prompt
    if retrieved_docs:
        full_prompt += "\nHere are some reference docs:"
        for i, doc in enumerate(retrieved_docs):
            full_prompt += "\nAPI " + str(i) + ": " + str(doc)
    
    return base_prompt, full_prompt


def get_response_llada(model, tokenizer, question, api_name, device, steps=128, gen_length=256, block_length=32, temperature=0., retrieved_docs=None):
    """Get response from LLaDA model for a single question.
    
    Returns:
        tuple: (response, system_prompt, user_prompt, retrieved_docs_list)
            - response: The model's generated response
            - system_prompt: The system prompt used
            - user_prompt: The full user prompt with context
            - retrieved_docs_list: List of retrieved doc strings for logging
    """
    
    # Convert Document objects to strings if needed
    retrieved_docs_list = None
    if retrieved_docs:
        retrieved_docs_list = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in retrieved_docs]
    
    # Encode the question (returns base_prompt and full_prompt with docs)
    base_prompt, full_prompt = encode_question(question, api_name, retrieved_docs=retrieved_docs_list)
    
    # Format for instruct model with system prompt (matching retriever version)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": full_prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    # Tokenize
    encoded = tokenizer(
        formatted_prompt,
        add_special_tokens=False,
        return_tensors="pt"
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Generate
    output = generate(
        model, 
        input_ids, 
        attention_mask, 
        steps=steps, 
        gen_length=gen_length, 
        block_length=block_length, 
        temperature=temperature, 
        cfg_scale=0., 
        remasking='low_confidence'
    )
    
    # Decode
    response = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    return response, SYSTEM_PROMPT, full_prompt, retrieved_docs_list


def write_result_to_file(result, output_file):
    """Write a single result to the output file."""
    with open(output_file, "a") as outfile:
        json.dump(result, outfile)
        outfile.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct", 
                        help="HuggingFace model name for LLaDA")
    parser.add_argument("--output_file", type=str, required=True, 
                        help="the output file this script writes to")
    parser.add_argument("--question_data", type=str, required=True, 
                        help="path to the questions data file")
    parser.add_argument("--api_name", type=str, required=True, 
                        help="API dataset name: 'torchhub', 'tensorhub', or 'huggingface'")
    parser.add_argument("--steps", type=int, default=128, 
                        help="Number of diffusion steps for LLaDA")
    parser.add_argument("--gen_length", type=int, default=128, 
                        help="Maximum generation length")
    parser.add_argument("--block_length", type=int, default=32, 
                        help="Block length for semi-autoregressive generation")
    parser.add_argument("--temperature", type=float, default=0., 
                        help="Sampling temperature (0 for deterministic)")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to run inference on")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size for inference (currently only 1 is supported)")
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Maximum number of questions to process (for testing, None for all)")
    parser.add_argument("--api_dataset", type=str, default=None,
                        help="Path to the API dataset for retrieval (enables RAG mode)")
    parser.add_argument("--num_docs", type=int, default=1,
                        help="Number of API docs to retrieve (when using retrieval, default=1 to match retriever version)")
    parser.add_argument("--retriever", type=str, default="bm25", choices=["bm25", "gpt"],
                        help="Which retriever to use: 'bm25' (local, fast) or 'gpt' (OpenAI embeddings, semantic)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key for GPT retriever (or set OPENAI_API_KEY env var)")
    args = parser.parse_args()

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading LLaDA model: {args.model_name}")
    model = AutoModel.from_pretrained(
        args.model_name, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, 
        trust_remote_code=True
    )
    
    # LLaDA works better with left padding
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    
    # Check that pad token doesn't equal mask token
    assert tokenizer.pad_token_id != 126336, "Pad token ID should not equal mask token ID"

    # Initialize retriever if API dataset is provided
    retriever = None
    if args.api_dataset:
        print(f"Loading API dataset for retrieval: {args.api_dataset}")
        print(f"Using {args.retriever} retriever with top {args.num_docs} docs")
        
        if args.retriever == "gpt":
            # GPT retriever uses OpenAI embeddings
            api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("GPT retriever requires OpenAI API key. Set --api_key or OPENAI_API_KEY env var.")
            os.environ["OPENAI_API_KEY"] = api_key
            
            retriever = GPTRetriever(index=[], query_kwargs={"similarity_top_k": args.num_docs})
            index_path = args.retriever + '_dataset_index.json'
            
            if os.path.exists(index_path):
                print(f"Loading existing GPT index from: {index_path}")
                retriever.load_from_disk(index_path)
            else:
                print(f"Creating GPT index (this may take a while)...")
                documents = JSONLReader().load_data(args.api_dataset)
                retriever.from_documents(documents)
                retriever.save_to_disk(retriever.index, index_path)
                print(f"Saved GPT index to: {index_path}")
                
        elif args.retriever == "bm25":
            # BM25 retriever uses term frequency
            corpus = []
            with open(args.api_dataset, 'r') as f:
                for line in f:
                    corpus.append(json.loads(line))
            tokenized_corpus = [str(doc).split(" ") for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            retriever = BM25Retriever(index=bm25, corpus=corpus, query_kwargs={"similarity_top_k": args.num_docs})
        
        print(f"Retriever initialized successfully")

    # Read questions
    print(f"Loading questions from: {args.question_data}")
    questions = []
    question_ids = []
    with open(args.question_data, 'r') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data["text"])
            question_ids.append(data["question_id"])
    
    print(f"Loaded {len(questions)} questions")

    # Limit questions if specified
    if args.max_questions is not None:
        questions = questions[:args.max_questions]
        question_ids = question_ids[:args.max_questions]
        print(f"Limited to {len(questions)} questions for testing")

    # Remove existing output file if it exists
    if os.path.exists(args.output_file):
        print(f"Removing existing output file: {args.output_file}")
        os.remove(args.output_file)

    # Process each question
    start_time = time.time()
    
    for idx, (question, question_id) in enumerate(tqdm(zip(questions, question_ids), total=len(questions), desc="Processing")):
        try:
            # Retrieve relevant API docs if retriever is available
            retrieved_docs = None
            if retriever is not None:
                retrieved_docs = retriever.get_relevant_documents(question)
            
            response, system_prompt, user_prompt, retrieved_docs_list = get_response_llada(
                model, 
                tokenizer, 
                question, 
                args.api_name,
                device,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                retrieved_docs=retrieved_docs
            )
            
            # Full result with context for analysis
            result = {
                'text': response,
                'question_id': question_id, 
                'answer_id': "None", 
                'model_id': args.model_name,
                # Additional context for analysis
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'original_question': question,
                'api_name': args.api_name,
                'retrieved_docs': retrieved_docs_list,
                'metadata': {
                    'steps': args.steps,
                    'gen_length': args.gen_length,
                    'block_length': args.block_length,
                    'temperature': args.temperature,
                    'retriever': args.retriever if retriever else None,
                    'num_docs': args.num_docs if retriever else None
                }
            }
            
        except Exception as e:
            print(f"Error processing question {question_id}: {e}")
            import traceback
            traceback.print_exc()
            result = {
                'text': f"Error: {str(e)}", 
                'question_id': question_id, 
                'answer_id': "None", 
                'model_id': args.model_name,
                'system_prompt': SYSTEM_PROMPT,
                'user_prompt': None,
                'original_question': question,
                'api_name': args.api_name,
                'retrieved_docs': None,
                'metadata': {'error': str(e)}
            }
        
        write_result_to_file(result, args.output_file)
        
        # Print progress every 10 questions
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            remaining = avg_time * (len(questions) - idx - 1)
            print(f"\nProgress: {idx + 1}/{len(questions)} | Avg time: {avg_time:.2f}s | ETA: {remaining/60:.1f} min")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n{'='*50}")
    print(f"Completed processing {len(questions)} questions")
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Average time per question: {elapsed_time/len(questions):.2f} seconds")
    print(f"Results saved to: {args.output_file}")


if __name__ == '__main__':
    main()
