using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Speech.Recognition;
using System.Speech.Synthesis;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using static System.Windows.Forms.VisualStyles.VisualStyleElement;

namespace questionAnswer
{
    public partial class Form1 : Form
    {
        SpeechRecognitionEngine sp1 = new SpeechRecognitionEngine();
        SpeechSynthesizer speaker = new SpeechSynthesizer();
        public Form1()
        {
            InitializeComponent();
            foreach (InstalledVoice voice in speaker.GetInstalledVoices())
            {
                comboBox1.Items.Add(voice.VoiceInfo.Name);

            }
        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void Form1_Load(object sender, EventArgs e)
        {
            button1.BackColor = Color.Green;
            sp1.SetInputToDefaultAudioDevice();
            Choices ch1 = new Choices();
            ch1.Add(new string[] { "What is the capital city of Azerbaijan", "How are you", "How is the weather today" });
            GrammarBuilder g = new GrammarBuilder();
            g.Append(ch1);
            Grammar grammar = new Grammar(g);
            sp1.LoadGrammarAsync(grammar);
            sp1.SpeechRecognized += Sp1_SpeechRecognized;
        }

        private void Sp1_SpeechRecognized(object sender, SpeechRecognizedEventArgs e)
        {
            richTextBox1.Text += e.Result.Text;

            switch (e.Result.Text.ToString())
            {
                case "How are you":
                    richTextBox2.Text = "I am good.";
                    break;
                case "What is the capital city of Azerbaijan":
                    richTextBox2.Text = "The capital city of Azerbaijan is Baku";
                    break;
                case "How is the weather today":
                    richTextBox2.Text = "The weather today is both windy and rainy";
                    break;
            }
            speaker.SpeakAsync(richTextBox2.Text);

        }
        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            speaker.SelectVoice(comboBox1.SelectedItem.ToString());
        }
        private void button1_Click(object sender, EventArgs e)
        {
            if (button1.BackColor == Color.Green)
            {
                speaker.SpeakAsyncCancelAll();
                richTextBox1.Clear();
                button1.BackColor = Color.Red;
                button1.Text = "Stop";
                sp1.RecognizeAsync(RecognizeMode.Multiple);
                
            }
            else
            {
                button1.BackColor = Color.Green;
                button1.Text = "Ask";
                sp1.RecognizeAsyncStop();



            }
        }

      
    }
}
