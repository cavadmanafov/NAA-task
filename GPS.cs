using GMap.NET;
using GMap.NET.MapProviders;
using GMap.NET.WindowsForms;
using GMap.NET.WindowsForms.Markers;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using static GMap.NET.Entity.OpenStreetMapGraphHopperGeocodeEntity;

namespace GPSprogram
{
    public partial class Form1 : Form
    {
        GMap.NET.WindowsForms.GMapControl gmap;
        double glat, glong;
        public Form1()
        {
            InitializeComponent();
            gmap = new GMap.NET.WindowsForms.GMapControl();

            gmap.MapProvider = GMap.NET.MapProviders.GMapProviders.GoogleMap;
            gmap.Dock = DockStyle.Fill;
            GMap.NET.GMaps.Instance.Mode = GMap.NET.AccessMode.ServerAndCache;
            gmap.MinZoom = 1;
            gmap.MaxZoom = 20;
            panel1.Controls.Add(gmap);
        }

        private void label4_Click(object sender, EventArgs e)
        {

        }

        private void Form1_Load(object sender, EventArgs e)
        {
            gmap.MapProvider = GMapProviders.GoogleMap;
            gmap.Position = new PointLatLng(40.71427, -74.00597);
            gmap.MinZoom = 1;

            gmap.MaxZoom = 18;

            gmap.Zoom = 13;
        }

        void gmap_mousemove(object sender, MouseEventArgs e)
        {
            textBox5.Text = e.X.ToString();

            textBox4.Text = e.Y.ToString();

            var point = gmap.FromLocalToLatLng(e.X, e.Y);
            double lat = point.Lat;
            double lon = point.Lng;
           
            label6.Text = lat.ToString();
            label7.Text = lon.ToString();



        }

        private void button3_Click(object sender, EventArgs e)
        {
            gmap.MouseMove += gmap_mousemove;
            gmap.MouseClick += gmap_mouseclick;
        }
        void gmap_mouseclick(object sender, MouseEventArgs e)
        {
            var point = gmap.FromLocalToLatLng(e.X, e.Y);
            double lat = point.Lat;
            double lon = point.Lng;
            glat = lat;
            glong = lon;
            label6.Text = lat.ToString();
            label7.Text = lon.ToString();
            MessageBox.Show(glat.ToString() + " " + glong.ToString());
        }
        private void AddMarker(double lat, double lng, string title) 
        {
            GMapMarker marker = new GMarkerGoogle(new PointLatLng(lat, lng), GMarkerGoogleType.red); 

            marker.ToolTipText = title; 

            GMapOverlay markersOverlay = new GMapOverlay("markers"); 

            markersOverlay.Markers.Add(marker);
            gmap.Overlays.Add(markersOverlay); 
        }

        private void panel1_Paint(object sender, PaintEventArgs e)
        {

        }

        private void trackBar1_ValueChanged(object sender, EventArgs e)
        {
            gmap.Zoom = trackBar1.Value;
        }

        private void button2_Click(object sender, EventArgs e)
        {
            MessageBox.Show(glat.ToString() + " " + glong.ToString());
             AddMarker(Convert.ToDouble(glat), Convert.ToDouble(glong), textBox3.Text); 
           // AddMarker(0, 0, textBox3.Text);

        }
    }
}
