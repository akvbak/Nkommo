import gradio as gr
from bot import stream_response

with gr.Blocks() as demo:
    gr.Markdown("# UG Buddy")
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone","upload"], type="filepath", label="Upload Audio")
        text_input  = gr.Textbox(label="Or enter text message")
    lang_dropdown = gr.Dropdown(
        choices=["English", "Twi", "Ga", "Ewe", "Hausa", "Dagbani"],
        value="English",
        label="Select your language"
    )
    chatbot = gr.Chatbot()
    state   = gr.State([])

    gr.Button("Send Audio").click(
        fn=stream_response,
        inputs=[audio_input, state, gr.State(True), lang_dropdown],
        outputs=[chatbot, state]
    )
    gr.Button("Send Text").click(
        fn=stream_response,
        inputs=[text_input, state, gr.State(False), lang_dropdown],
        outputs=[chatbot, state]
    )

demo.launch(share=True)