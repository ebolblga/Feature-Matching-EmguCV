﻿namespace AI_Lab1._3;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;

public partial class Form1 : Form
{
    List<Image<Gray, byte>> SearchImages = new List<Image<Gray, byte>>();
    public Form1()
    {
        this.InitializeComponent();
    }

    // Feature matching two images
    private void button1_Click(object sender, EventArgs e)
    {
        try
        {
            if (SearchImages.Count < 1)
            {
                MessageBox.Show("Select a serach folder", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            using (OpenFileDialog ofd = new OpenFileDialog() { Multiselect = false, Filter = "Image Files (*.jpg;*.png;*.bmp;)|*.jpg;*.png;*.bmp;|All Files (*.*)|*.*;" })
            {
                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    int maxIndex = 0;
                    int maxCount = 0;
                    Image<Bgr, byte> inputImage = new Image<Bgr, byte>(ofd.FileName);

                    for (int i = 0; i < SearchImages.Count - 1; i++)
                    {
                        Tuple<VectorOfPoint, int> tmp = ProcessImage(SearchImages[i], inputImage.Convert<Gray, byte>());
                        if (tmp.Item2 > maxCount)
                        {
                            maxIndex = i;
                            maxCount = tmp.Item2;
                        }
                    }

                    Tuple<VectorOfPoint, int> vp = ProcessImage(SearchImages[maxIndex], inputImage.Convert<Gray, byte>());

                    if (vp != null)
                    {
                        // CvInvoke.Polylines(SearchImages[maxIndex], vp.Item1, true, new MCvScalar(0, 0, 255), 5);
                    }
                    else
                    {
                        MessageBox.Show("Did not find a match", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        return;
                    }

                    pictureBox2.Image = SearchImages[maxIndex].AsBitmap();
                    pictureBox1.Image = inputImage.AsBitmap();
                }
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show(ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }

    // Exit button
    private void exitToolStripMenuItem_Click(object sender, EventArgs e)
    {
        this.Close();
    }

    // Load folder of jpegs
    private void button2_Click(object sender, EventArgs e)
    {
        using (var fbd = new FolderBrowserDialog())
        {
            DialogResult result = fbd.ShowDialog();

            if (result == DialogResult.OK && !string.IsNullOrWhiteSpace(fbd.SelectedPath))
            {
                SearchImages.Clear();

                foreach (string file in Directory.EnumerateFiles(fbd.SelectedPath, "*.jpg"))
                {
                    // Opens image as bitmap from path
                    Bitmap img = (Bitmap)Image.FromFile(file);

                    Image<Bgr, byte> inputImage = new Image<Bgr, byte>(file);

                    SearchImages.Add(inputImage.Convert<Gray, byte>());
                    inputImage.Dispose();
                }
            }
        }
    }

    // Processing
    private static Tuple<VectorOfPoint, int> ProcessImage(Image<Gray, byte> matchImg, Image<Gray, byte> inputImg)
    {
        try
        {
            VectorOfPoint finalPoints = null;
            Mat homography = null;
            VectorOfKeyPoint matchImgKeyPoints = new VectorOfKeyPoint();
            VectorOfKeyPoint inputImgKeyPoints = new VectorOfKeyPoint();
            Mat matchImgDescriptor = new Mat();
            Mat inputImgDescriptor = new Mat();
            Mat mask;
            int k = 2;
            double uniquenessThreshold = 0.80;
            VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();


            // Feature detection and description
            Brisk featureDetector = new Brisk();
            featureDetector.DetectAndCompute(matchImg, null, matchImgKeyPoints, matchImgDescriptor, false);
            featureDetector.DetectAndCompute(inputImg, null, inputImgKeyPoints, inputImgDescriptor, false);

            // Matching
            BFMatcher matcher = new BFMatcher(DistanceType.Hamming);
            matcher.Add(matchImgDescriptor);
            matcher.KnnMatch(inputImgDescriptor, matches, k);

            mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
            mask.SetTo(new MCvScalar(255));

            Features2DToolbox.VoteForUniqueness(matches, uniquenessThreshold, mask);

            int matchCount = Features2DToolbox.VoteForSizeAndOrientation(matchImgKeyPoints, inputImgKeyPoints, matches, mask, 1.5, 20);
            if (matchCount >= 4)
            {
                homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(matchImgKeyPoints, inputImgKeyPoints, matches, mask, 5);
            }

            if (homography != null)
            {
                Rectangle rect = new Rectangle(Point.Empty, matchImg.Size);
                PointF[] pts = new PointF[]
                {
                        new PointF(rect.Left,rect.Bottom),
                        new PointF(rect.Right,rect.Bottom),
                        new PointF(rect.Right,rect.Top),
                        new PointF(rect.Left,rect.Top)
                };

                pts = CvInvoke.PerspectiveTransform(pts, homography);
                Point[] points = Array.ConvertAll<PointF, Point>(pts, Point.Round);
                finalPoints = new VectorOfPoint(points);
            }

            //return Tuple<VectorOfPoint?, int?>(finalPoints, matchCount);
            return Tuple.Create(finalPoints, matchCount);
        }
        catch(Exception ex)
        {
            throw new Exception(ex.Message);
        }
    }
}