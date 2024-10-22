using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;  // for File I/O
using System.Globalization;  // for converting strings to floats
using System.Linq;
using System;

class Job
{
    public Job(Color _color, int _blockindex, GameObject prefab, 
        Vector3 _position, Vector3 _source, Vector3 _sink, float[] _timetable)
    {
        
        // Instantiate는 static 메서드이므로 Object.Instantiate로 호출해야 합니다.
        block = UnityEngine.Object.Instantiate(prefab);
        // 생성한 인스턴스에서 Block 컴포넌트 가져오기
        Block blockComp = block.GetComponent<Block>();
        // 초기화 메서드 호출
        blockComp.Initialize(_color, _blockindex, _source, _position, _sink, _timetable);
        Debug.Log("Job" + _blockindex + " Created!");
        // block.SetActive(false);
    }

    public void Activate()
    {
        block.SetActive(true);
    }

    public GameObject block;

}
class Machine
{
    public Machine(GameObject prefab, Vector3 initialPosition)
    {

        // Instantiate는 static 메서드이므로 Object.Instantiate로 호출해야 합니다.
        process = UnityEngine.Object.Instantiate(prefab);
        // 생성한 인스턴스에서 Block 컴포넌트 가져오기
        Process processComp = process.GetComponent<Process>();
        processComp.transform.position = initialPosition;

        process.SetActive(true);
    }
    public GameObject process;

}




public class GameManager : MonoBehaviour
{
    public GameObject BlockPrefab;
    public GameObject ProcessPrefab;


    public string colorPath = "color.csv"; // CSV 파일 경로 (프로젝트 폴더 내)
    public string positionPath = "position.csv";
    public string timePath = "time_ATCS.csv";
    public int hFactor = 0;
    private List<Color> colorData; // CSV 파일로부터 읽은 RGB 값들 저장
    private List<Vector3> positionData;
    private List<(int idx, int machine, float release, float move, float setup, float start, float finish, int jobSetup, int machineSetup)> timeData;
    public Transform parent;
    static int numBlocks = 100;
    static int numMachine = 5;
    static int numProcesses = 1;
    private int sink_idx;
    public bool isfinished;

    // Job 객체 배열 선언
    private Job[] jobs;  // Job[] 타입으로 선언
    private Machine[] machines;
    private Vector3 pos;
    private Vector3 processposition;
    private Vector3 machineposition;
    int count;
    List<Vector3> source;
    List<Vector3> sink;
    private float timer = 0.0f;
    int num_created;
    // Start is called before the first frame update
    void Start()
    {
        Time.timeScale = 10f;
        // 1. CSV 파일 읽기
        colorData = ReadColorsFromCSV(colorPath);
        positionData = ReadPositionFromCSV(positionPath);
        timeData = ReadTimeTableFromCSV(timePath);
        num_created = 0;
        sink_idx = 0;
        // Job 배열 크기 지정
        jobs = new Job[numBlocks];  // 3개의 Job을 담을 수 있는 배열 생성
        machines = new Machine[numMachine];

        source = stackPositions(10, 0.0f, 0.0f, 1.0f, 0.6f, -0.6f);
        sink = stackPositions(10, 11.4f, 0.0f, 6.4f, -0.6f, -0.6f);
        
        // 각각의 Job 객체 생성
        for (int d = 0; d < numBlocks; d++)
        {
            int coloridx = d % colorData.Count;
            int machineidx = d % numMachine;
            Debug.Log("-------------------------------------------");
            Debug.Log(d);
            Debug.Log("Machine: "+d);
            Debug.Log(timeData[d]);
            Debug.Log(positionData[timeData[d].Item2]);

            float[] timetable = new float[] { timeData[d].Item3, timeData[d].Item4, timeData[d].Item5, timeData[d].Item6, timeData[d].Item7 };
            jobs[d] = new Job(colorData[timeData[d].Item8], d, BlockPrefab,
            positionData[timeData[d].Item2], source[d], sink[d], timetable);
        }

            
        for (int j = 0; j < numMachine; j++)
        {
            Debug.Log(j);
            machineposition = new Vector3(positionData[j].x, -0.305f, positionData[j].z);
            machines[j] = new Machine(ProcessPrefab, machineposition);
            Debug.Log("New machine " + j + " generated on " + machineposition);
        }
        
    }

    // Update is called once per frame
    void Update()
    {
        
        isfinished = check_termination();
        if (isfinished)
        {
            return;
        }

        if (timer >= timeData[num_created].Item3)
        {
            jobs[num_created].Activate();
            if (num_created <numBlocks - 1)
            {
                num_created++;
            }
        }
        timer += Time.deltaTime;

    }

    bool check_termination()
    {

        int count = 0;
        
        for (int i = 0; i < jobs.Length; i++)
        {
            if (jobs[i].block.GetComponent<Block>().isFinished == true)
            {
                if (!jobs[i].block.GetComponent<Block>().isSinkUpdated)
                {
                    jobs[i].block.GetComponent<Block>().SetSink(sink[sink_idx]);
                    sink_idx++;
                    jobs[i].block.GetComponent<Block>().isSinkUpdated = true;
                    Debug.Log("Block " + jobs[i].block.GetComponent<Block>().blockindex + " is assigned Sink " + sink_idx);
                }
                
                count++;

            }
        }

        if (count == jobs.Length)
        {
            Debug.Log("All Jobs Finished!");
            return true;
        }
        else
        {
            return false;
        }
    }
    List<(int idx, int machine, float release, float move, float setup, float start, float finish, int jobSetup, int machineSetup)> ReadTimeTableFromCSV(string file)
    {
        var log = new List<(int idx, int machine, float release, float move, float setup, float start, float finish, int jobSetup, int machineSetup)>();

        string path = Path.Combine(Application.dataPath, file);  // 파일 경로

        if (File.Exists(path))
        {
            string[] lines = File.ReadAllLines(path);  // 모든 라인을 읽음

            foreach (string line in lines.Skip(1))
            {
                string[] values = line.Split(',');  // 쉼표로 값들을 분리
                int idx = int.Parse(values[0], CultureInfo.InvariantCulture); 
                int machine = int.Parse(values[1], CultureInfo.InvariantCulture); 
                float release = float.Parse(values[2], CultureInfo.InvariantCulture); 
                float move = float.Parse(values[3], CultureInfo.InvariantCulture);  
                // float setup = float.Parse(values[4], CultureInfo.InvariantCulture);
                float setup = float.Parse(values[3], CultureInfo.InvariantCulture) + 2.0f;
                float start = float.Parse(values[5], CultureInfo.InvariantCulture); 
                float finish = float.Parse(values[6], CultureInfo.InvariantCulture);
                int jobSetup = int.Parse(values[7], CultureInfo.InvariantCulture);
                int machineSetup = int.Parse(values[8], CultureInfo.InvariantCulture);

                // ValueTuple을 리스트에 추가
                log.Add((idx, machine, release, move, setup, start, finish, jobSetup, machineSetup));
            }

        }
        else
        {
            Debug.LogError("CSV file not found at: " + path);
        }

        return log;

    }
    // CSV 파일에서 RGB 값을 읽어와서 List<float[3]> 형태로 저장
    List<Vector3> ReadPositionFromCSV(string file)
    {
        List<Vector3> positions = new List<Vector3>();  // 색상 리스트
        string path = Path.Combine(Application.dataPath, file);  // 파일 경로
        if (File.Exists(path))
        {
            string[] lines = File.ReadAllLines(path);  // 모든 라인을 읽음

            foreach (string line in lines)
            {
                string[] values = line.Split(',');  // 쉼표로 값들을 분리
                float x = float.Parse(values[0], CultureInfo.InvariantCulture);  // Red
                float y = float.Parse(values[1], CultureInfo.InvariantCulture);  // Green
                float z = float.Parse(values[2], CultureInfo.InvariantCulture) - 12.0f * hFactor;  // Blue

                positions.Add(new Vector3(x, y, z));  // float[3]로 저장
            }
        }
        else
        {
            Debug.LogError("CSV file not found at: " + path);
        }

        return positions;
    }

    // CSV 파일에서 RGB 값을 읽어와서 List<float[3]> 형태로 저장
    List<Color> ReadColorsFromCSV(string file)
    {
        List<Color> colors = new List<Color>();  // 색상 리스트
        string path = Path.Combine(Application.dataPath, file);  // 파일 경로

        if (File.Exists(path))
        {
            string[] lines = File.ReadAllLines(path);  // 모든 라인을 읽음

            foreach (string line in lines)
            {
                string[] values = line.Split(',');  // 쉼표로 값들을 분리
                float r = float.Parse(values[1], CultureInfo.InvariantCulture);  // Red
                float g = float.Parse(values[2], CultureInfo.InvariantCulture);  // Green
                float b = float.Parse(values[3], CultureInfo.InvariantCulture);  // Blue

                colors.Add(new Color(r / 255f, g / 255f, b / 255f));  // float[3]로 저장
            }
        }
        else
        {
            Debug.LogError("CSV file not found at: " + path);
        }

        return colors;
    }

    List<Vector3> stackPositions(int n_row, float startX, float startY, float startZ, float zOffset, float xOffset)
    {
        
        List<Vector3> sources = new List<Vector3>();
        // 처음 3개의 블록은 z 방향으로 쌓고, 그 이후로는 x 방향으로 나가면서 z 방향으로 쌓음
        int a;
        int b;
        for (int i = 0; i < numBlocks; i++)
        {
            a = i / n_row;
            b = i % n_row;
            sources.Add(new Vector3(startX + xOffset * a, startY, startZ - 12.0f * hFactor + zOffset * b));
            
        }
        return sources;
    }
}
