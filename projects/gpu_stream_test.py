from langchain_ollama import ChatOllama


def main() -> None:
    llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.7)
    print("Testing of GPU....")
    prompt = (
        "I want to learn ethical hacking where can i start i have newly installed arch linux and i have basic knowledge of linux and programming?"
    )

    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)

    print("\n\n--- Test Complete ---")


if __name__ == "__main__":
    main()
