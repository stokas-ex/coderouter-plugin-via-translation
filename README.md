# coderouter-plugin-via-translation

CodeRouter の `translate` adapter で、会話を次の順に処理します。

```text
自然言語 A
  → 入力翻訳モデル（A → B）
  → 本命モデル（B で処理）
  → 出力翻訳モデル（B → A）
  → 自然言語 A
```

翻訳モデルはそれぞれ独立に OpenAI 互換 API の URL とモデル名で選べます。本命モデルは通常の CodeRouter provider と同じ `base_url` / `model` で設定します。翻訳プロンプトは個別に指定でき、本命モデルのシステムプロンプトには CodeRouter 標準の `append_system_prompt` を使えます。

## インストール

```bash
uv pip install ./coderouter-plugin-via-translation
```

CodeRouter 2.8.0 以降が必要です。プラグインは adapter entry point として登録されます。インストール後、`providers.yaml` の `plugins.enabled` に `translate` を明示して有効化してください。

## 設定

[`examples/providers.translation.yaml`](./examples/providers.translation.yaml) を `~/.coderouter/providers.yaml` に合わせて設定します。

```yaml
plugins:
  enabled: [translate]
  config:
    translate:
      language_a: Japanese
      language_b: English
      input_translator:
        base_url: http://127.0.0.1:8081/v1
        model: translator-ja-en
        api_key_env: TRANSLATE_IN_API_KEY
      output_translator:
        base_url: http://127.0.0.1:8082/v1
        model: translator-en-ja
        api_key_env: TRANSLATE_OUT_API_KEY
      input_system_prompt: >-
        Translate from {source_language} to {target_language}. Return only
        the translation and preserve code and formatting.
      output_system_prompt: >-
        Translate from {source_language} to {target_language}. Return only
        the translation and preserve code and formatting.

providers:
  - name: translated-main
    kind: translate
    base_url: http://127.0.0.1:8083/v1
    model: main-model
    append_system_prompt: "Answer in English."

profiles:
  - name: translated
    providers: [translated-main]
default_profile: translated
```

`base_url` は OpenAI 互換 API の `/v1` までを指定します。キーは YAML に直接書かず、必要な場合だけ `api_key_env` に環境変数名を指定してください。`input_system_prompt` と `output_system_prompt` では `{source_language}` / `{target_language}`（または `{language_a}` / `{language_b}`）を使えます。本命モデルへの追加指示は `append_system_prompt` に言語 B で記述します。

## 動作と制約

- テキストの user / assistant 履歴を言語 B に翻訳します。system メッセージと tool 結果は変更せず、tool call の構造も保持します。
- 本命モデルの最終テキストを言語 A に翻訳します。tool call を含む stream も構造を維持します。
- 出力翻訳には回答全体が必要なため、stream は本命モデルの生成完了までバッファされます。翻訳が終わるまで最初の応答チャンクは送られません。
- 翻訳 API の障害は provider error として明示されます。誤った言語のまま成功したように返すフォールバックは行いません。
- 3 モデル間で会話テキストが送信されるため、各モデル endpoint のデータ取り扱いを確認してください。

## 開発・テスト

```bash
uv pip install -e ".[dev]"
python -m pytest -q
```

## ライセンス

MIT
