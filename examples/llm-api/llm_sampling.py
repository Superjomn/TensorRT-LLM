### :title Sampling Techniques Showcase
### :order 6
### :section Customization
"""
This example demonstrates various sampling techniques available in TensorRT-LLM.
It showcases different sampling parameters and their effects on text generation.
"""

from typing import Optional

import click

from tensorrt_llm import LLM, SamplingParams

# Example prompts to demonstrate different sampling techniques
prompts = [
    "What is the future of artificial intelligence?",
    "Describe a beautiful sunset over the ocean.",
    "Write a short story about a robot discovering emotions.",
]


def demonstrate_greedy_decoding(llm: LLM, prompt: str):
    """Demonstrates greedy decoding with temperature=0."""
    print("\n🎯 === GREEDY DECODING ===")
    print("Using temperature=0 for deterministic, focused output")

    sampling_params = SamplingParams(
        max_tokens=50,
        temperature=0.0,  # Greedy decoding
    )

    response = llm.generate(prompt, sampling_params)
    print(f"Prompt: {prompt}")
    print(f"Response: {response.outputs[0].text}")


def demonstrate_temperature_sampling(llm: LLM, prompt: str):
    """Demonstrates temperature sampling with different temperature values."""
    print("\n🌡️ === TEMPERATURE SAMPLING ===")
    print(
        "Higher temperature = more creative/random, Lower temperature = more focused"
    )

    temperatures = [0.3, 0.7, 1.0, 1.5]

    for temp in temperatures:
        sampling_params = SamplingParams(
            max_tokens=50,
            temperature=temp,
        )

        response = llm.generate(prompt, sampling_params)
        print(f"Temperature {temp}: {response.outputs[0].text}")


def demonstrate_top_k_sampling(llm: LLM, prompt: str):
    """Demonstrates top-k sampling with different k values."""
    print("\n🔝 === TOP-K SAMPLING ===")
    print("Only consider the top-k most likely tokens at each step")

    top_k_values = [1, 5, 20, 50]

    for k in top_k_values:
        sampling_params = SamplingParams(
            max_tokens=50,
            temperature=0.8,  # Use moderate temperature
            top_k=k,
        )

        response = llm.generate(prompt, sampling_params)
        print(f"Top-k {k}: {response.outputs[0].text}")


def demonstrate_top_p_sampling(llm: LLM, prompt: str):
    """Demonstrates top-p (nucleus) sampling with different p values."""
    print("\n🎯 === TOP-P (NUCLEUS) SAMPLING ===")
    print("Only consider tokens whose cumulative probability is within top-p")

    top_p_values = [0.1, 0.5, 0.9, 0.95]

    for p in top_p_values:
        sampling_params = SamplingParams(
            max_tokens=50,
            temperature=0.8,  # Use moderate temperature
            top_p=p,
        )

        response = llm.generate(prompt, sampling_params)
        print(f"Top-p {p}: {response.outputs[0].text}")


def demonstrate_combined_sampling(llm: LLM, prompt: str):
    """Demonstrates combined top-k and top-p sampling."""
    print("\n🔄 === COMBINED TOP-K + TOP-P SAMPLING ===")
    print("Using both top-k and top-p together for balanced control")

    sampling_params = SamplingParams(
        max_tokens=50,
        temperature=0.8,
        top_k=40,  # Consider top 40 tokens
        top_p=0.9,  # Within 90% cumulative probability
    )

    response = llm.generate(prompt, sampling_params)
    print(f"Combined (k=40, p=0.9): {response.outputs[0].text}")


def demonstrate_multiple_sequences(llm: LLM, prompt: str):
    """Demonstrates generating multiple sequences with different sampling."""
    print("\n📚 === MULTIPLE SEQUENCES ===")
    print("Generate multiple different responses for the same prompt")

    sampling_params = SamplingParams(
        max_tokens=40,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        n=3,  # Generate 3 different sequences
    )

    response = llm.generate(prompt, sampling_params)
    print(f"Prompt: {prompt}")
    for i, output in enumerate(response.outputs):
        print(f"Sequence {i+1}: {output.text}")


def demonstrate_beam_search(llm: LLM, prompt: str):
    """Demonstrates beam search for high-quality outputs."""
    print("\n🔍 === BEAM SEARCH ===")
    print("Explore multiple paths simultaneously for higher quality")

    sampling_params = SamplingParams(
        max_tokens=50,
        use_beam_search=True,
        n=3,  # Return top 3 beam search results
    )

    response = llm.generate(prompt, sampling_params)
    print(f"Prompt: {prompt}")
    for i, output in enumerate(response.outputs):
        print(f"Beam {i+1}: {output.text}")


def demonstrate_with_logprobs(llm: LLM, prompt: str):
    """Demonstrates generation with log probabilities."""
    print("\n📊 === GENERATION WITH LOG PROBABILITIES ===")
    print("Get probability information for generated tokens")

    sampling_params = SamplingParams(
        max_tokens=20,
        temperature=0.7,
        top_k=50,
        logprobs=True,  # Return log probabilities
    )

    response = llm.generate(prompt, sampling_params)
    output = response.outputs[0]

    print(f"Prompt: {prompt}")
    print(f"Generated: {output.text}")

    if hasattr(output, 'logprobs') and output.logprobs:
        print("Token log probabilities:")
        for i, logprob_info in enumerate(output.logprobs):
            if logprob_info:  # Check if logprob_info is not None
                token_id = list(logprob_info.keys())[0]
                logprob = list(logprob_info.values())[0]
                print(f"  Token {i}: ID={token_id}, LogProb={logprob:.4f}")


def run_all_demonstrations(model_path: Optional[str] = None):
    """Run all sampling demonstrations."""
    print("🚀 TensorRT-LLM Sampling Techniques Showcase")
    print("=" * 50)

    # Initialize LLM
    llm = LLM(model=model_path or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
              # You can change this to any model you have available
              )

    # Use the first prompt for most demonstrations
    demo_prompt = prompts[0]

    # Run all demonstrations
    demonstrate_greedy_decoding(llm, demo_prompt)
    demonstrate_temperature_sampling(llm, demo_prompt)
    demonstrate_top_k_sampling(llm, demo_prompt)
    demonstrate_top_p_sampling(llm, demo_prompt)
    demonstrate_combined_sampling(llm, demo_prompt)
    # TODO[Superjomn]: enable this once pytorch backend supports multiple sequences
    # demonstrate_multiple_sequences(llm, demo_prompt)
    demonstrate_beam_search(llm, demo_prompt)
    demonstrate_with_logprobs(llm, demo_prompt)

    print("\n🎉 All sampling demonstrations completed!")


@click.command()
@click.option("--model",
              type=str,
              default=None,
              help="Path to the model or model name")
@click.option("--demo",
              type=click.Choice([
                  "greedy", "temperature", "top_k", "top_p", "combined",
                  "multiple", "beam", "logprobs", "creative", "all"
              ]),
              default="all",
              help="Which demonstration to run")
@click.option("--prompt", type=str, default=None, help="Custom prompt to use")
def main(model: Optional[str], demo: str, prompt: Optional[str]):
    """
    Showcase various sampling techniques in TensorRT-LLM.

    Examples:
        python llm_sampling.py --demo all
        python llm_sampling.py --demo temperature --prompt "Tell me a joke"
        python llm_sampling.py --demo beam --model path/to/your/model
    """

    # Initialize LLM
    llm = LLM(model=model or "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
              # You can change this to any model you have available
              )

    demo_prompt = prompt or prompts[0]

    # Run specific demonstration
    if demo == "greedy":
        demonstrate_greedy_decoding(llm, demo_prompt)
    elif demo == "temperature":
        demonstrate_temperature_sampling(llm, demo_prompt)
    elif demo == "top_k":
        demonstrate_top_k_sampling(llm, demo_prompt)
    elif demo == "top_p":
        demonstrate_top_p_sampling(llm, demo_prompt)
    elif demo == "combined":
        demonstrate_combined_sampling(llm, demo_prompt)
    elif demo == "multiple":
        demonstrate_multiple_sequences(llm, demo_prompt)
    elif demo == "beam":
        demonstrate_beam_search(llm, demo_prompt)
    elif demo == "logprobs":
        demonstrate_with_logprobs(llm, demo_prompt)
    elif demo == "all":
        run_all_demonstrations(model)


if __name__ == "__main__":
    main()
