<head>

<?php
echo "<span style='font-size:30px'>AI新聞可信度辨識結果<br></span>";
echo "<br>";

echo "loading python... </br>... </br>...";
/*
$command = escapeshellcmd('python3 /home/yi-hsien/ntnu/NTNU_GIMC_FakeNewsDetector/demo/demo.py ');
$output = shell_exec($command.$_POST[title].' '.$_POST[content]);
echo $output;
*/
$param1 = $_POST['title'];
$param2 = $_POST['content'];






$path = "python3 /home/yi-hsien/ntnu/NTNU_GIMC_FakeNewsDetector/demo/demo_2.py ";
//passthru($path.$param1.' '.$param2);






error_reporting(E_ALL); 
ini_set('display_errors', 1);

exec($path.$param1.' '.$param2.' 2>&1', $output);

foreach ($output as $value)
    echo $value


?>
<meta http-equiv="Content-Type" content="text/html" charset="utf-8″>
</head>
