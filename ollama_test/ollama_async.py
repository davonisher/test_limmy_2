import asyncio
from ollama import AsyncClient

async def translate_text(text, client, model="mistral"):
    try:
        response = await client.generate(
            model=model,
            prompt=f"translate this French text into English: {text}"
        )
        return response['response'].lstrip(), response['total_duration']
    except Exception as e:
        print(f"Oops! Translation error: {e}")
        return None, 0

async def process_batch(texts, host="http://127.0.0.1:11434"):
    client = AsyncClient(host=host)
    tasks = [translate_text(text, client) for text in texts]
    return await asyncio.gather(*tasks)

async def main():
    reports = ["report_text_1", "report_text_2", "report_text_3", "report_text_4"]
    
    # Process all reports in parallel
    results = await process_batch(reports)
    
    for i, (translation, duration) in enumerate(results):
        if translation:
            print(f"Report {i+1} translated in {duration}ms")
            print(f"Translation: {translation[:100]}...\n")

if __name__ == "__main__":
    asyncio.run(main())