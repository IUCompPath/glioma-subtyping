IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]
kaiko_mean = [0.5, 0.5, 0.5]
kaiko_std =  [0.5, 0.5, 0.5]
optimus_mean = [0.707223, 0.578729, 0.703617]
optimus_std =[0.211883, 0.230117, 0.177517]
hibou_mean=[0.7068, 0.5755, 0.7220]
hibou_std=[0.1950, 0.2316, 0.1816]
lunit_mean=[ 0.70322989, 0.53606487, 0.66096631 ]
lunit_std=[ 0.21716536, 0.26081574, 0.20723464 ]

MODEL2CONSTANTS = {
	"resnet": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    "simclr": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    "hipt4k": {
		"mean": kaiko_mean,
		"std": kaiko_std
	},
    "retccl": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    
    "ctranspath": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    "lunit": {
		"mean": lunit_mean,
		"std": lunit_std
	},
	"uni":
	{
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    "uni2":
	{
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    "conchv1_5":
	{
		"mean": OPENAI_MEAN,
		"std": OPENAI_STD
	},
	"gigapath":
	{
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},

	"conch_v1":
	{
		"mean": OPENAI_MEAN,
		"std": OPENAI_STD
	},
	"optimus":
	{
		"mean": optimus_mean,
		"std": optimus_std
	},
	"optimus1":
	{
		"mean": optimus_mean,
		"std": optimus_std
	},
	"hibou":
	{
		"mean": hibou_mean,
		"std": hibou_std
	},
    "ssl_ours":
	{
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    "ssl_ours_reg":
	{
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    "ssl_dinov3":
    {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	}
}