#include <iostream>
#include "cv.hpp"
#include <string>

using namespace std;
using namespace cv;

#define NofImgs 10

int main() {
	//필요변수 선언
	string query_name;
	string imgName;
	vector<String> DBs_filename;
	Mat query, descriptor_qry, DBs_imgs[NofImgs], descriptor_DBs[NofImgs];
	Ptr<ORB> orbF = ORB::create(1000);
	vector<KeyPoint> keypoints_qry, keypoints_DBs[NofImgs];
	vector<vector<DMatch>> matches[NofImgs]; //descriptor match
	vector<DMatch> goodMatches[NofImgs];
	BFMatcher matcher(NORM_HAMMING);
	Mat imgMatches[NofImgs];
	int i, k = 2, DBs_nofImgs, index = -1;
	float nndr = 0.6f;


	//사용자에게서 query image 이름 입력 받기
	cout << "Enter query image name: ";
	cin >> query_name;
	imgName = "query_image/" + query_name;
	cout << imgName << endl;
	waitKey(1000);
	//DB와 query에서 각각 imgs 읽어오기
	query = imread(imgName);
	glob("DBs", DBs_filename, false);

	DBs_nofImgs = DBs_filename.size();

	cout << "Sample Image Load Size: " << DBs_nofImgs << endl;

	for (i = 0; i < DBs_nofImgs; i++) {
		DBs_imgs[i] = imread(DBs_filename[i]);
		//없는 경우 예외 처리
		if (query.empty() || DBs_imgs[i].empty()) {
			cout << "No file!" << endl;
			return -1;
		}
		resize(DBs_imgs[i], DBs_imgs[i], Size(640, 480));
	}
	resize(query, query, Size(640, 480));

	//query img와 DB imgs들의 feature, descriptor추출 및 생성
	orbF->detectAndCompute(query, noArray(), keypoints_qry, descriptor_qry);

	for (i = 0; i < DBs_nofImgs; i++) {
		orbF->detectAndCompute(DBs_imgs[i], noArray(), keypoints_DBs[i], descriptor_DBs[i]);

		//1st, 2nd pair 만들기
		matcher.knnMatch(descriptor_qry, descriptor_DBs[i], matches[i], k);
	}

	//NNDR계산해서 good matching인지 판정
	for (int j = 0; j < DBs_nofImgs; j++) {
		for (i = 0; i < matches[j].size(); i++) {
			if (matches[j].at(i).size() == 2 && matches[j].at(i).at(0).distance <= nndr * matches[j].at(i).at(1).distance)
				goodMatches[j].push_back(matches[j][i][0]);
		}
	}

	//good matching 시각화하기
	for (i = 0; i < DBs_nofImgs; i++) {
		drawMatches(query, keypoints_qry, DBs_imgs[i], keypoints_DBs[i], goodMatches[i], imgMatches[i], Scalar::all(-1), Scalar(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		cout << "image number " << i + 1 << " Matching: " << goodMatches[i+1].size() << endl;

		if (goodMatches[i].size() > 5)
			index = i;
	}

	imshow("Query", query);
	if (index != -1) {
		cout << "Match found!" << endl;

		imshow("Best_matching", imgMatches[index]);
		waitKey(0);
	}
	else
		cout << "Match Not found!" << endl;

	return 0;
}
