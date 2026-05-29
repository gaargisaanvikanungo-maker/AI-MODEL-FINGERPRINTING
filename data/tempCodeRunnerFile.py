
        except Exception as e:
            print(f"  ERROR — {model_name} prompt {row['prompt_id']}: {e}")
            results.append({
                "prompt_id"    : r