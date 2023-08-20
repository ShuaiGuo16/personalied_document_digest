# Content: Class definition of interactive chatbot system 
# Author: Shuai Guo
# Email: shuai.guo@ch.abb.com
# Date: August, 2023

import azure.cognitiveservices.speech as speechsdk
from pydub import AudioSegment
from langchain.llms import OpenAI, AzureOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os



class AudioEngine:
    """Create audio for the interview script."""

    def __init__(self, scripts, voice_names, article_name, themes, audience, LLM_engine):
        """Specify embedding model.

        Args:
        --------------
        scripts: dict, interview scripts.
        voice_names: dict, voice names employed for the journalist & author bots.
        article_name: name of the article associated with the interview.
        themes: theme of the article.
        audience: target audiences.
        LLM_engine: the embedding model. 
                For a complete list of supported embedding models in LangChain, 
                see https://python.langchain.com/docs/integrations/text_embedding/
        """

        # Extract interview scripts
        self.Q_scripts = scripts['questions']
        self.A_scripts = scripts['answers']
        self.article_name = article_name
        self.themes = themes
        self.audience = audience

        # Extract voice types
        self.Q_voice = voice_names['questions']
        self.A_voice = voice_names['answers']

        # Config speech
        self.speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), 
                                                    region=os.environ.get('SPEECH_REGION'))
        
        # Instantiate an LLM (for creating intro speech)
        if LLM_engine == 'OpenAI':
            self.llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0.8
            )
        
        elif LLM_engine == 'Azure':
            self.llm = AzureOpenAI(
                    deployment_name="deployment-5af509f3323342ee919481751c6f8b7d",
                    model_name="text-davinci-003",
                    openai_api_base="https://abb-chcrc.openai.azure.com/",
                    openai_api_version="2023-03-15-preview",
                    openai_api_key=os.environ["OPENAI_API_KEY_AZURE"],
                    openai_api_type="azure",
                )

        else:
            raise KeyError("Currently unsupported chat model type!")



    def synthesize_speech(self, verbose=True):
        """Synthesize the speech for the entire interview script.

        Args:
        --------------
        verbose: report synthesize status.
        """   

        # Instantiate an empty audio handler
        combined_audio = AudioSegment.empty()

        for idx, (question, answer) in enumerate(zip(self.Q_scripts, self.A_scripts)):
            if idx == 0:
                self._create_introductory_speech(verbose)
                intro_audio = AudioSegment.from_wav(f"./speech/intro_"+self.article_name+".wav")

            Q_filename = f"./speech/question_{idx}_"+self.article_name+".wav"
            A_filename = f"./speech/answer_{idx}_"+self.article_name+".wav"

            self._synthesize_single_speech(question, self.Q_voice, Q_filename, verbose)
            self._synthesize_single_speech(answer, self.A_voice, A_filename, verbose)

            Q_audio = AudioSegment.from_wav(Q_filename)
            A_audio = AudioSegment.from_wav(A_filename)

            if idx == 0:
                combined_audio += intro_audio + Q_audio + A_audio
            else:
                combined_audio += Q_audio + A_audio

        # Save combined audio to a single file
        combined_filename = "combined_conversation_"+self.article_name+".wav"
        combined_audio.export(combined_filename, format="wav")



    def _create_introductory_speech(self, verbose):
        """Convert introductory text into speech.

        Args:
        --------
        verbose: report synthesize status.
        """

        # Introductory text
        text = self._create_introductory_text()

        # Synthesize speech
        intro_filename = f"./speech/intro_"+self.article_name+".wav"
        self._synthesize_single_speech(text, self.Q_voice, intro_filename, verbose)




    def _synthesize_single_speech(self, text, voice, filename, verbose=True):
        """Synthesize a single speaker speech.

        Args:
        --------------
        text: text to convert.
        voice: name of the voice.
        filename: location to store the generated audio file.
        verbose: report synthesize status.
        """

        # Change voice
        self.speech_config.speech_synthesis_voice_name = voice

        # Set up audio output
        audio_config = speechsdk.audio.AudioOutputConfig(filename=filename)
        
        # Synthesize speech
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, 
                                                        audio_config=audio_config)
        result = speech_synthesizer.speak_text_async(text).get()

        # Status report
        if verbose:
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                print(f"Speech synthesized to file {filename}.")
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                print("Speech synthesis canceled: {}".format(cancellation_details.reason))
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    if cancellation_details.error_details:
                        print("Error details: {}".format(cancellation_details.error_details))
                        print("Did you set the speech resource key and region values?")




    def _create_introductory_text(self):
        """Introductory text for the interview.

        Outputs:
        --------
        response: the created introductory texts.
        """

        # Prompt engineering
        template = """Based on the theme "{theme}", the target audience of "{audience}", and the 
        questions posed during the interview, craft a short introductory text for the interview in the tone of a journalist. 
        The introductory text should set the stage for the interview and align with the essence of the conversation.
        The introductory text should mention the company name ABB.

        [Question History]: {question_list}
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["theme", "audience", "question_list"],
        )

        # LLM inference
        response = self.llm.predict(prompt.format(theme=self.themes, audience=self.audience,
                                            question_list=self.Q_scripts))
        
        return response