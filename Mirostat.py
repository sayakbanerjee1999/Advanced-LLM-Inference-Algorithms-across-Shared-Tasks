import torch
import math
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

def mirostat(model, tokenizer, prompt, max_length=50, device='cpu', temperature=1.0, target_ce=3.0, learning_rate=0.1):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    mu = 2 * target_ce  # Initial mu value / "maximal surprisal"
    # target_ce = target surprise value = \tau in the paper

    k_list = []
    s_list = []
    mu_list = []
    error_list = []
    surprisal_list = []
    logit_distribution = {}


    for step in range(max_length):
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]
            adjusted_logits = logits / temperature
            adjusted_probs = torch.softmax(adjusted_logits, dim=-1)

            sorted_logits, sorted_inds = torch.sort(adjusted_logits, descending = True)

        # if step+1 in [1, 10, 100]:
        #     logit_distribution[step+1] = sorted_logits

        m = 100
        i = torch.arange(1, m, device=device, dtype=torch.float32)
        t_i = torch.log((i+1)/i)                    # i ranges from 1->(m-1); Num->(i+1), Deno(i)

        # b_i = log(p(i)/p(i+1))    i -> 1-(m-1) or (0-(m-2))
        top_m_probs = torch.softmax(sorted_logits[0, :m], dim = -1)
        b_i = torch.log(top_m_probs[: m-1] / top_m_probs[1 : m])

        s_hat_num = torch.sum(t_i * b_i, dim = -1)
        s_hat_deno = torch.sum(b_i**2)
        s_hat = (s_hat_num / s_hat_deno).item() 

        # Compute k using Zipf exponent
        vocab_size = sorted_logits.shape[-1]
        epsilon_hat = s_hat - 1
        k_numerator = epsilon_hat * (2 ** mu)
        k_denominator = (1 - (m**(-epsilon_hat)))
        # Map Sure k is an integer between 1 and vocab_size
        k = max(1, min((k_numerator / k_denominator)**(1/s_hat), vocab_size))
        k = int(k)

        # top k sampling
        topk_logits = sorted_logits[:,0:k]
        topk_inds = sorted_inds[:,0:k]
        topk_probs = torch.softmax(topk_logits, dim=1)
        next_tok = topk_inds[0, torch.multinomial(topk_probs, num_samples=1)]
        input_ids = torch.cat([input_ids, next_tok], dim=1)
        if next_tok.item() == tokenizer.eos_token_id:
            break

        # surprisal error and adjust mu accordingly
        next_token_problem = adjusted_probs[0, next_tok.item()]
        token_surprisal = -torch.log2(next_token_problem).item()
        err = token_surprisal - target_ce
        mu = mu - (learning_rate*err)

        # Update Lists
        s_list.append(s_hat)
        k_list.append(k)
        error_list.append(err)
        mu_list.append(mu)
        surprisal_list.append(token_surprisal)


    # Sequence Generated
    sequence = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    # List contains value corresponding to every token
    np_surprisal = np.array(surprisal_list)
    per_tkn_surprisal = 2**(np_surprisal)
    mean_surprisal = np.mean(per_tkn_surprisal)
    std_surprisal = np.std(per_tkn_surprisal)
    median_surprisal = np.median(per_tkn_surprisal)
    sequence_ppl = 2**(np.mean(np_surprisal))

    output_metrics = {
        "mean_surprisal_per_token": mean_surprisal,
        "std_surprisal_per_token": std_surprisal,
        "median_surprisal_per_token": median_surprisal,
        "suquence_perplexity": sequence_ppl
    }

    # OutPut Graph
    plt.figure(figsize=(10,4))
    plt.plot(k_list, label="k", color='blue')
    plt.xlabel("Generation step")
    plt.ylabel("k")
    plt.title(f"k vs Generation Step for τ={target_ce}")
    plt.grid(True)
    plt.show()

    # Plot s_list, mu_list, error_list
    plt.figure(figsize=(10,4))
    plt.plot(s_list, label="Zipf exponent", color='orange')
    plt.plot(mu_list, label="μ", color='green')
    plt.plot(error_list, label="Surprisal error", color='red')
    plt.xlabel("Generation step")
    plt.ylabel("Value")
    plt.title(f"s, mu, error vs Generation Step for τ={target_ce}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Output Graph for Logit Distribution
    # print(logit_distribution[10][0][:50].cpu().numpy().shape)
    # plt.figure(figsize=(10,4))

    # for step in [1, 10, 100]:
    #     logits = logit_distribution[step][0].cpu().numpy()
    #     plt.plot(logits, label=f"Step {step}")

    # plt.xlabel("Token index (sorted)")
    # plt.ylabel("Logit value")
    # plt.title("Logit Distributions at Different Steps")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return sequence, output_metrics


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt_list = ["The capital of France is,", "Once upon a time,"]
    tau_list = [2.5, 3, 4]
    for tau in tau_list:
        for prompt in prompt_list:
            seq, output_metrics = mirostat(model, tokenizer, prompt, max_length=128,
                                          device=device, temperature=0.9, target_ce=tau, learning_rate=0.1)
            print(f"""Parameters: \n Prompt:{prompt} \n tau:{tau}""")
            print(f"Generated Seq: {seq}")
            print(f"Output Metrics: {output_metrics}")
