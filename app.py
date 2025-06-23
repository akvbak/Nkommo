import gradio as gr
from bot import stream_response

with gr.Blocks(css=".gr-box {border-radius: 12px;}") as demo:
    gr.Markdown(
        """
        # 🤖 Nkommo v1
        _Talk to your AI assistant in English or any supported Ghanaian language._
        """, 
        elem_id="title"
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="🎤 Upload or Record Audio"
            )
            send_audio_btn = gr.Button("🎧 Send Audio", variant="primary")
        
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="💬 Or type your message",
                placeholder="Type your message here..."
            )
            send_text_btn = gr.Button("📩 Send Text", variant="primary")

    lang_dropdown = gr.Dropdown(
        choices=["English", "Twi", "Ga", "Ewe", "Hausa", "Dagbani"],
        value="English",
        label="🌍 Select Language"
    )

    chatbot = gr.Chatbot(label="🗨️ Chat History")
    state = gr.State([])

    send_audio_btn.click(
        fn=stream_response,
        inputs=[audio_input, state, gr.State(True), lang_dropdown],
        outputs=[chatbot, state]
    )

    send_text_btn.click(
        fn=stream_response,
        inputs=[text_input, state, gr.State(False), lang_dropdown],
        outputs=[chatbot, state]
    )

demo.launch(share=True)
