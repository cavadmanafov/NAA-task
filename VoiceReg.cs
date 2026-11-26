using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Speech.Recognition;
namespace VoiceRecognition
{
    public partial class Form1 : Form
    {
        SpeechRecognitionEngine sp1 = new SpeechRecognitionEngine();
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            button1.BackColor = Color.Green;
            sp1.SetInputToDefaultAudioDevice();
            Choices ch1 = new Choices();
            ch1.Add(new string[] {"test", "hello", "text"});
            GrammarBuilder g = new GrammarBuilder();
            g.Append(ch1);
            Grammar grammar = new Grammar(g);
            sp1.LoadGrammarAsync(grammar);
            sp1.SpeechRecognized += Sp1_SpeechRecognized;
        }

        private void Sp1_SpeechRecognized(object sender, SpeechRecognizedEventArgs e)
        {
            richTextBox1.Text += "\n" + e.Result.Text;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if(button1.BackColor == Color.Green)
            {
                button1.BackColor = Color.Red;
                button1.Text = "Stop";
                sp1.RecognizeAsync(RecognizeMode.Multiple);
            }
            else
            {
                button1.BackColor = Color.Green;
                button1.Text = "Start";
                sp1.RecognizeAsyncStop();
               
            }
        }
    }
}
