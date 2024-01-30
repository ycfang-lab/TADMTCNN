import torch
import torchvision
import copy

#===============================================================================
#梯度反转层,由task_discrimator调用
class GradientReversal(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg()*ctx.alpha
        return output, None

#===============================================================================
#将reshape操作变为一个module，好嵌入到torch.nn.Sequential中去,由task_discrimator调用
class Reshape(torch.nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()
    
    def forward(self,x):
        return x.permute(0,2,1)

#===============================================================================
#任务判别器
class task_discrimator(torch.nn.Module):
    def __init__(self, num_task, level=1):
        super(task_discrimator, self).__init__()
        if level == 1:
            self._task_discrimator = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d((1,1)),
                torch.nn.Flatten(-2, -1),
                Reshape(),
                torch.nn.AdaptiveAvgPool1d(128),
                torch.nn.Flatten(1),
                torch.nn.Linear(128, num_task)
            )
        elif level == 2:
            self._task_discrimator = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d((1,1)),
                torch.nn.Flatten(-2, -1),
                Reshape(),
                torch.nn.AdaptiveAvgPool1d(128),
                torch.nn.Flatten(1),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(32, num_task)
            )

    def forward(self, x, beta=None):
        if beta:
            x = GradientReversal.apply(x, beta)
        return self._task_discrimator(x)

#===============================================================================
#AdvFocusDMTL
class AdvFocusDMTL(torch.nn.Module):
    def __init__(self, tasks, backbone="resnet18", shared_dis=False, pretrained=False, alpha_grain=1, discrimator_level=1, input_size=100):
        super(AdvFocusDMTL, self).__init__()
        self.tasks = tasks
        self.shared_dis = shared_dis
        if backbone == "resnet18":
            resnet = torchvision.models.resnet18(pretrained=pretrained)
        elif backbone == "resnet50":
            resnet = torchvision.models.resnet50(pretrained=pretrained)
        elif backbone == "resnet101":
            resnet = torchvision.models.resnet101(pretrained=pretrained)
        elif backbone == "resnet152":
            resnet = torchvision.models.resnet152(pretrained=pretrained)
        self.subnets = torch.nn.ModuleDict({task:copy.deepcopy(resnet) for task in self.tasks})
        self.share_net = copy.deepcopy(resnet)
        self.avgpool = resnet.avgpool

        if shared_dis:
            self.task_discrimator = task_discrimator(len(self.tasks), discrimator_level)
        else:
            self.task_discrimator = torch.nn.ModuleList([task_discrimator(len(self.tasks), discrimator_level) for i in range(4)])

        if alpha_grain == 1:
            alpha = {task:torch.nn.Parameter(torch.tensor([0.5])).requires_grad_() for task in ['self', 'shared']}
            alpha_sha = {task:torch.nn.Parameter(torch.tensor([0.5])).requires_grad_() for task in self.tasks + ['shared']}
            layer_alpha = {task:copy.deepcopy(torch.nn.ParameterDict(alpha)) for task in tasks} # 浅拷贝？
            layer_alpha['shared'] = torch.nn.ParameterDict(alpha_sha)
            self.layer_alpha = torch.nn.ModuleList([torch.nn.ModuleDict(copy.deepcopy(layer_alpha)) for i in range(4)]) 

        elif alpha_grain == 2:
            temp_x = torch.randn((1,3,input_size, input_size))
            temp_x = resnet.conv1(temp_x)
            temp_x = resnet.maxpool(temp_x)

            temp_x = resnet.layer1(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in ['self', 'shared']}
            alpha_sha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in self.tasks + ['shared']}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer_alpha['shared'] = torch.nn.ParameterDict(alpha_sha)
            layer1_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            temp_x = resnet.layer2(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in ['self', 'shared']}
            alpha_sha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in self.tasks + ['shared']}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer_alpha['shared'] = torch.nn.ParameterDict(alpha_sha)
            layer2_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            temp_x = resnet.layer3(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in ['self', 'shared']}
            alpha_sha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in self.tasks + ['shared']}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer_alpha['shared'] = torch.nn.ParameterDict(alpha_sha)
            layer3_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            temp_x = resnet.layer4(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in ['self', 'shared']}
            alpha_sha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in self.tasks + ['shared']}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer_alpha['shared'] = torch.nn.ParameterDict(alpha_sha)
            layer4_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            self.layer_alpha = torch.nn.ModuleList([layer1_alpha, layer2_alpha, layer3_alpha, layer4_alpha])

        elif alpha_grain == 3:
            temp_x = torch.randn((1,3,input_size, input_size))
            temp_x = resnet.conv1(temp_x)
            temp_x = resnet.maxpool(temp_x)

            temp_x = resnet.layer1(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in ['self', 'shared']}
            alpha_sha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in self.tasks + ['shared']}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer_alpha['shared'] = torch.nn.ParameterDict(alpha_sha)
            layer1_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            temp_x = resnet.layer2(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in ['self', 'shared']}
            alpha_sha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in self.tasks + ['shared']}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer_alpha['shared'] = torch.nn.ParameterDict(alpha_sha)
            layer2_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            temp_x = resnet.layer3(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in ['self', 'shared']}
            alpha_sha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in self.tasks + ['shared']}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer_alpha['shared'] = torch.nn.ParameterDict(alpha_sha)
            layer3_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            temp_x = resnet.layer4(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in ['self', 'shared']}
            alpha_sha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in self.tasks + ['shared']}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer_alpha['shared'] = torch.nn.ParameterDict(alpha_sha)
            layer4_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            self.layer_alpha = torch.nn.ModuleList([layer1_alpha, layer2_alpha, layer3_alpha, layer4_alpha])

    def forward(self, inputs, fusion_loc="after", beta=None, pre_training = False):
        if not pre_training:
            if not isinstance(inputs, dict):
                dis_out = {task:[] for task in self.tasks}
                dis_out_share = []

                x = {task:self.subnets[task].conv1(inputs) for task in self.tasks}
                x = {task:self.subnets[task].bn1(x[task]) for task in self.tasks}
                x = {task:self.subnets[task].relu(x[task]) for task in self.tasks}
                x = {task:self.subnets[task].maxpool(x[task]) for task in self.tasks}

                share_x = self.share_net.conv1(inputs) 
                share_x = self.share_net.bn1(share_x) 
                share_x = self.share_net.relu(share_x) 
                share_x = self.share_net.maxpool(share_x) 

                x = {task:self.subnets[task].layer1(x[task]) for task in self.tasks}
                share_x = self.share_net.layer1(share_x)
                if fusion_loc == "before":
                    x, share_x = self._feature_fusion(x, share_x, self.layer_alpha[0])
                for task in self.tasks: dis_out[task].append(self.task_discrimator(x[task]) if self.shared_dis else self.task_discrimator[0](x[task]))
                dis_out_share.append(self.task_discrimator(share_x) if self.shared_dis else self.task_discrimator[0](share_x))
                if fusion_loc == "after":
                    x, share_x = self._feature_fusion(x, share_x, self.layer_alpha[0])

                x = {task:self.subnets[task].layer2(x[task]) for task in self.tasks}
                share_x = self.share_net.layer2(share_x)
                if fusion_loc == "before":
                    x, share_x = self._feature_fusion(x, share_x, self.layer_alpha[1])
                for task in self.tasks: dis_out[task].append(self.task_discrimator(x[task]) if self.shared_dis else self.task_discrimator[1](x[task]))
                dis_out_share.append(self.task_discrimator(share_x) if self.shared_dis else self.task_discrimator[1](share_x))
                if fusion_loc == "after":
                    x, share_x = self._feature_fusion(x, share_x, self.layer_alpha[1])

                x = {task:self.subnets[task].layer3(x[task]) for task in self.tasks}
                share_x = self.share_net.layer3(share_x)
                if fusion_loc == "before":
                    x, share_x = self._feature_fusion(x, share_x, self.layer_alpha[2])
                for task in self.tasks: dis_out[task].append(self.task_discrimator(x[task]) if self.shared_dis else self.task_discrimator[2](x[task]))
                dis_out_share.append(self.task_discrimator(share_x) if self.shared_dis else self.task_discrimator[2](share_x))
                if fusion_loc == "after":
                    x, share_x = self._feature_fusion(x, share_x, self.layer_alpha[2])

                x = {task:self.subnets[task].layer4(x[task]) for task in self.tasks}
                share_x = self.share_net.layer4(share_x)
                if fusion_loc == "before":
                    x,_ = self._feature_fusion(x, share_x, self.layer_alpha[3])             
                for task in self.tasks: dis_out[task].append(self.task_discrimator(x[task]) if self.shared_dis else self.task_discrimator[3](x[task]))
                dis_out_share.append(self.task_discrimator(share_x) if self.shared_dis else self.task_discrimator[3](share_x))
                if fusion_loc == "after":
                    x,_ = self._feature_fusion(x, share_x, self.layer_alpha[3])
        
                x = {task:self.avgpool(x[task]) for task in self.tasks}
                subnet_out = {task:x[task].flatten(1) for task in self.tasks}

                return subnet_out, dis_out, dis_out_share

            else:
                dis_out = {task:[] for task in self.tasks}
                dis_out_shared = {task:[] for task in self.tasks}
                subnet_out = {}
                for task in inputs.keys():
                    x = inputs[task]
                    x = self.subnets[task].conv1(inputs) 
                    x = self.subnets[task].bn1(x) 
                    x = self.subnets[task].relu(x) 
                    x = self.subnets[task].maxpool(x)
                    x = self.subnets[task].layer1(x)
                    share_x = inputs[task]
                    share_x = self.share_net.conv1(inputs) 
                    share_x = self.share_net.bn1(share_x) 
                    share_x = self.share_net.relu(share_x) 
                    share_x = self.share_net.maxpool(share_x)

                    share_x = self.share_nets.layer1(share_x)
                    if fusion_loc == "before":
                        x,share_x = self._feature_fusion({task:x}, share_x, self.layer_alpha[0]) 
                    dis_out[task].append(self.task_discrimator(x[task]) if self.shared_dis else self.task_discrimator[0](x))
                    dis_out_shared[task].append(self.task_discrimator(x[task], beta) if self.shared_dis else self.task_discrimator[0](share_x, beta))
                    if fusion_loc == "after":
                        x,share_x = self._feature_fusion({task:x}, share_x, self.layer_alpha[0])

                    x = self.subnets[task].layer2(x)
                    share_x = self.share_net.layer2(share_x)
                    if fusion_loc == "before":
                        x,share_x = self._feature_fusion({task:x}, share_x, self.layer_alpha[1])
                    dis_out[task].append(self.task_discrimator(x[task]) if self.shared_dis else self.task_discrimator[1](x))
                    dis_out_shared[task].append(self.task_discrimator(x[task], beta) if self.shared_dis else self.task_discrimator[1](share_x, beta))
                    if fusion_loc == "after":
                        x,share_x = self._feature_fusion({task:x}, share_x, self.layer_alpha[1])

                    x = self.subnets[task].layer3(x)
                    share_x = self.share_net.layer3(share_x)
                    if fusion_loc == "before":
                        x,share_x = self._feature_fusion({task:x}, share_x, self.layer_alpha[2])
                    dis_out[task].append(self.task_discrimator(x[task]) if self.shared_dis else self.task_discrimator[2](x))
                    dis_out_shared[task].append(self.task_discrimator(x[task], beta) if self.shared_dis else self.task_discrimator[2](share_x, beta))
                    if fusion_loc == "after":
                        x,share_x = self._feature_fusion({task:x}, share_x, self.layer_alpha[2])

                    x = self.subnets[task].layer4(x)
                    share_x = self.share_net.layer4(share_x)
                    if fusion_loc == "before":
                        x,share_x = self._feature_fusion({task:x}, share_x, self.layer_alpha[3])
                    dis_out[task].append(self.task_discrimator(x[task]) if self.shared_dis else self.task_discrimator[3](x))
                    dis_out_shared[task].append(self.task_discrimator(x[task], beta) if self.shared_dis else self.task_discrimator[3](share_x, beta))
                    if fusion_loc == "after":
                        x,share_x = self._feature_fusion({task:x}, share_x, self.layer_alpha[3])

                    output = self.avgpool(x[task])
                    subnet_out[task] = output.flatten(1)
                
                return subnet_out, dis_out, dis_out_shared
        else:
            if not isinstance(inputs, dict):
                dis_out = {task:[] for task in self.tasks}
                dis_out_share = []

                x = {task:self.subnets[task].conv1(inputs) for task in self.tasks}
                x = {task:self.subnets[task].bn1(x[task]) for task in self.tasks}
                x = {task:self.subnets[task].relu(x[task]) for task in self.tasks}
                x = {task:self.subnets[task].maxpool(x[task]) for task in self.tasks}

                share_x = self.share_net.conv1(inputs) 
                share_x = self.share_net.bn1(share_x) 
                share_x = self.share_net.relu(share_x) 
                share_x = self.share_net.maxpool(share_x) 

                x = {task:self.subnets[task].layer1(x[task]) for task in self.tasks}
                share_x = self.share_net.layer1(share_x)


                x = {task:self.subnets[task].layer2(x[task]) for task in self.tasks}
                share_x = self.share_net.layer2(share_x)


                x = {task:self.subnets[task].layer3(x[task]) for task in self.tasks}
                share_x = self.share_net.layer3(share_x)

                x = {task:self.subnets[task].layer4(x[task]) for task in self.tasks}
                share_x = self.share_net.layer4(share_x)

        
                x = {task:self.avgpool(x[task]) for task in self.tasks}
                subnet_out = {task:x[task].flatten(1) for task in self.tasks}

                share_x = self.avgpool(share_x)
                share_out = share_x.flatten(1)

                return subnet_out, share_out

            else:
                dis_out = {task:[] for task in self.tasks}
                dis_out_shared = {task:[] for task in self.tasks}
                subnet_out = {}
                for task in inputs.keys():
                    x = inputs[task]
                    x = self.subnets[task].conv1(inputs) 
                    x = self.subnets[task].bn1(x) 
                    x = self.subnets[task].relu(x) 
                    x = self.subnets[task].maxpool(x)
                    x = self.subnets[task].layer1(x)
                    share_x = inputs[task]
                    share_x = self.share_net.conv1(inputs) 
                    share_x = self.share_net.bn1(share_x) 
                    share_x = self.share_net.relu(share_x) 
                    share_x = self.share_net.maxpool(share_x)

                    share_x = self.share_nets.layer1(share_x)
                    if fusion_loc == "before":
                        x,share_x = self._feature_fusion({task:x}, share_x, self.layer_alpha[0]) 
                    dis_out[task].append(self.task_discrimator(x[task]) if self.shared_dis else self.task_discrimator[0](x))
                    dis_out_shared[task].append(self.task_discrimator(x[task], beta) if self.shared_dis else self.task_discrimator[0](share_x, beta))
                    if fusion_loc == "after":
                        x,share_x = self._feature_fusion({task:x}, share_x, self.layer_alpha[0])

                    x = self.subnets[task].layer2(x)
                    share_x = self.share_net.layer2(share_x)
                    if fusion_loc == "before":
                        x,share_x = self._feature_fusion({task:x}, share_x, self.layer_alpha[1])
                    dis_out[task].append(self.task_discrimator(x[task]) if self.shared_dis else self.task_discrimator[1](x))
                    dis_out_shared[task].append(self.task_discrimator(x[task], beta) if self.shared_dis else self.task_discrimator[1](share_x, beta))
                    if fusion_loc == "after":
                        x,share_x = self._feature_fusion({task:x}, share_x, self.layer_alpha[1])

                    x = self.subnets[task].layer3(x)
                    share_x = self.share_net.layer3(share_x)
                    if fusion_loc == "before":
                        x,share_x = self._feature_fusion({task:x}, share_x, self.layer_alpha[2])
                    dis_out[task].append(self.task_discrimator(x[task]) if self.shared_dis else self.task_discrimator[2](x))
                    dis_out_shared[task].append(self.task_discrimator(x[task], beta) if self.shared_dis else self.task_discrimator[2](share_x, beta))
                    if fusion_loc == "after":
                        x,share_x = self._feature_fusion({task:x}, share_x, self.layer_alpha[2])

                    x = self.subnets[task].layer4(x)
                    share_x = self.share_net.layer4(share_x)
                    if fusion_loc == "before":
                        x,share_x = self._feature_fusion({task:x}, share_x, self.layer_alpha[3])
                    dis_out[task].append(self.task_discrimator(x[task]) if self.shared_dis else self.task_discrimator[3](x))
                    dis_out_shared[task].append(self.task_discrimator(x[task], beta) if self.shared_dis else self.task_discrimator[3](share_x, beta))
                    if fusion_loc == "after":
                        x,share_x = self._feature_fusion({task:x}, share_x, self.layer_alpha[3])

                    output = self.avgpool(x[task])
                    subnet_out[task] = output.flatten(1)
                
                return subnet_out, dis_out, dis_out_shared

        

    def _feature_fusion(self, x, share_x, alpha):
        fused = {}
        temp = [share_x*alpha['shared']['shared']]
        for task in x.keys():
            fused[task] = x[task]*torch.sigmoid(alpha[task]['self']) + share_x*torch.sigmoid(alpha[task]['shared'])
            temp.append(x[task]*alpha['shared'][task])
        share_x = sum(temp) # 梯度断开？
        if len(x) == 1:
            task = list(x.keys())[0]
            return fused[task], share_x
        else:
            return fused, share_x 

#===============================================================================
#FocusDMTL
class FocusDMTL(torch.nn.Module):
    def __init__(self, tasks, backbone="resnet18", shared_dis=False, pretrained=False, alpha_grain=1, discrimator_level=1, input_size=100):
        super(FocusDMTL, self).__init__()
        self.tasks = tasks
        self.shared_dis = shared_dis
        if backbone == "resnet18":
            resnet = torchvision.models.resnet18(pretrained=pretrained)
        elif backbone == "resnet50":
            resnet = torchvision.models.resnet50(pretrained=pretrained)
        elif backbone == "resnet101":
            resnet = torchvision.models.resnet50(pretrained=pretrained)
        elif backbone == "resnet152":
            resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.subnets = torch.nn.ModuleDict({task:copy.deepcopy(resnet) for task in self.tasks})
        self.avgpool = resnet.avgpool

        if shared_dis:
            self.task_discrimator = task_discrimator(len(self.tasks), discrimator_level)
        else:
            self.task_discrimator = torch.nn.ModuleList([task_discrimator(len(self.tasks), discrimator_level) for i in range(4)])

        if alpha_grain == 1:
            alpha = {task:torch.nn.Parameter(torch.tensor([0.5])).requires_grad_() for task in self.tasks}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            self.layer_alpha = torch.nn.ModuleList([torch.nn.ModuleDict(copy.deepcopy(layer_alpha)) for i in range(4)])

        elif alpha_grain == 2:
            temp_x = torch.randn((1,3,input_size, input_size))
            temp_x = resnet.conv1(temp_x)
            temp_x = resnet.maxpool(temp_x)

            temp_x = resnet.layer1(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in self.tasks}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer1_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            temp_x = resnet.layer2(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in self.tasks}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer2_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            temp_x = resnet.layer3(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in self.tasks}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer3_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))
        
            temp_x = resnet.layer4(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in self.tasks}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer4_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            self.layer_alpha = torch.nn.ModuleList([layer1_alpha, layer2_alpha, layer3_alpha, layer4_alpha])

        elif alpha_grain == 3:
            temp_x = torch.randn((1,3,input_size, input_size))
            temp_x = resnet.conv1(temp_x)
            temp_x = resnet.maxpool(temp_x)

            temp_x = resnet.layer1(temp_x)            
            alpha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in self.tasks}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer1_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            temp_x = resnet.layer2(temp_x)            
            alpha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in self.tasks}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer2_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            temp_x = resnet.layer3(temp_x)            
            alpha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in self.tasks}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer3_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            temp_x = resnet.layer4(temp_x)            
            alpha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in self.tasks}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer4_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            self.layer_alpha = torch.nn.ModuleList([layer1_alpha, layer2_alpha, layer3_alpha, layer4_alpha])

    def forward(self, inputs, fusion_loc="after"):
        if not isinstance(inputs, dict):
            inputs = {task:inputs for task in self.tasks}
        dis_out = {task:[] for task in self.tasks}
        x = {task:self.subnets[task].conv1(inputs[task]) for task in self.tasks}
        x = {task:self.subnets[task].bn1(x[task]) for task in self.tasks}
        x = {task:self.subnets[task].relu(x[task]) for task in self.tasks}
        x = {task:self.subnets[task].maxpool(x[task]) for task in self.tasks}

        x = {task:self.subnets[task].layer1(x[task]) for task in self.tasks}
        if fusion_loc == "before":
            x = self._feature_fusion(x, self.layer_alpha[0])
        for task in self.tasks: dis_out[task].append(self.task_discrimator(x[task]) if self.shared_dis else self.task_discrimator[0](x[task]))
        if fusion_loc == "after":
            x = self._feature_fusion(x, self.layer_alpha[0])

        x = {task:self.subnets[task].layer2(x[task]) for task in self.tasks}
        if fusion_loc == "before":
            x = self._feature_fusion(x, self.layer_alpha[1])
        for task in self.tasks: dis_out[task].append(self.task_discrimator(x[task]) if self.shared_dis else self.task_discrimator[1](x[task]))
        if fusion_loc == "after":
            x = self._feature_fusion(x, self.layer_alpha[1])
        
        x = {task:self.subnets[task].layer3(x[task]) for task in self.tasks}
        if fusion_loc == "before":
            x = self._feature_fusion(x, self.layer_alpha[2])
        for task in self.tasks: dis_out[task].append(self.task_discrimator(x[task]) if self.shared_dis else self.task_discrimator[2](x[task]))
        if fusion_loc == "after":
            x = self._feature_fusion(x, self.layer_alpha[2])

        x = {task:self.subnets[task].layer4(x[task]) for task in self.tasks}
        if fusion_loc == "before":
            x = self._feature_fusion(x, self.layer_alpha[3])
        for task in self.tasks: dis_out[task].append(self.task_discrimator(x[task]) if self.shared_dis else self.task_discrimator[3](x[task]))
        if fusion_loc == "after":
            x = self._feature_fusion(x, self.layer_alpha[3])

        x = {task:self.avgpool(x[task]).flatten(1) for task in self.tasks}

        return x, dis_out
        
    def _feature_fusion(self, x, alpha):
        fused = {}
        for from_task in self.tasks:
            temp = []
            for to_task in self.tasks:
                temp.append(x[to_task]*torch.sigmoid(alpha[from_task][to_task]))
            fused[from_task] = sum(temp)
        return fused


#===============================================================================
#DMTL
class DMTL(torch.nn.Module):
    def __init__(self, tasks, backbone="resnet18", pretrained=False, alpha_grain=1, input_size=100):
        super(DMTL, self).__init__()
        self.tasks = tasks
        if backbone == "resnet18":
            resnet = torchvision.models.resnet18(pretrained=pretrained)
        elif backbone == "resnet50":
            resnet = torchvision.models.resnet50(pretrained=pretrained)
        elif backbone == "resnet101":
            resnet = torchvision.models.resnet50(pretrained=pretrained)
        elif backbone == "resnet152":
            resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.subnets = torch.nn.ModuleDict({task:copy.deepcopy(resnet) for task in self.tasks})
        self.avgpool = resnet.avgpool

        if alpha_grain == 1:
            alpha = {task:torch.nn.Parameter(torch.tensor([0.5])).requires_grad_() for task in self.tasks}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            self.layer_alpha = torch.nn.ModuleList([torch.nn.ModuleDict(copy.deepcopy(layer_alpha)) for i in range(4)])

        elif alpha_grain == 2:
            temp_x = torch.randn((1,3,input_size, input_size))
            temp_x = resnet.conv1(temp_x)
            temp_x = resnet.maxpool(temp_x)

            temp_x = resnet.layer1(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in self.tasks}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer1_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            temp_x = resnet.layer2(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in self.tasks}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer2_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            temp_x = resnet.layer3(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in self.tasks}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer3_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))
        
            temp_x = resnet.layer4(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in self.tasks}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer4_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            self.layer_alpha = torch.nn.ModuleList([layer1_alpha, layer2_alpha, layer3_alpha, layer4_alpha])

        elif alpha_grain == 3:
            temp_x = torch.randn((1,3,input_size, input_size))
            temp_x = resnet.conv1(temp_x)
            temp_x = resnet.maxpool(temp_x)

            temp_x = resnet.layer1(temp_x)            
            alpha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in self.tasks}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer1_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            temp_x = resnet.layer2(temp_x)            
            alpha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in self.tasks}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer2_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            temp_x = resnet.layer3(temp_x)            
            alpha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in self.tasks}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer3_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            temp_x = resnet.layer4(temp_x)            
            alpha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in self.tasks}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer4_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            self.layer_alpha = torch.nn.ModuleList([layer1_alpha, layer2_alpha, layer3_alpha, layer4_alpha])

    def forward(self, inputs):
        if not isinstance(inputs, dict):
            inputs = {task:inputs for task in self.tasks}
        x = {task:self.subnets[task].conv1(inputs[task]) for task in self.tasks}
        x = {task:self.subnets[task].bn1(x[task]) for task in self.tasks}
        x = {task:self.subnets[task].relu(x[task]) for task in self.tasks}
        x = {task:self.subnets[task].maxpool(x[task]) for task in self.tasks}

        x = {task:self.subnets[task].layer1(x[task]) for task in self.tasks}
        x = self._feature_fusion(x, self.layer_alpha[0])

        x = {task:self.subnets[task].layer2(x[task]) for task in self.tasks}
        x = self._feature_fusion(x, self.layer_alpha[1])
        
        x = {task:self.subnets[task].layer3(x[task]) for task in self.tasks}
        x = self._feature_fusion(x, self.layer_alpha[2])

        x = {task:self.subnets[task].layer4(x[task]) for task in self.tasks}
        x = self._feature_fusion(x, self.layer_alpha[3])

        x = {task:self.avgpool(x[task]).flatten(1) for task in self.tasks}

        return x
        
    def _feature_fusion(self, x, alpha, mode='sigmoid'):
        fused = {}
        for from_task in self.tasks:
            temp = []
            for to_task in self.tasks:
                if mode == 'sigmoid':
                    temp.append(x[to_task]*torch.sigmoid(alpha[from_task][to_task]))
                elif mode == 'linear':
                    temp.append(x[to_task]*alpha[from_task][to_task])
                elif mode == 'reg':
                    if from_task != to_task:
                        temp.append(x[to_task]*torch.sigmoid(alpha[from_task][to_task]))
                    temp.append(x[from_task])
            fused[from_task] = sum(temp)
        return fused

#===============================================================================
#ASPMTL
class ASPMTL(torch.nn.Module):
    def __init__(self, tasks, backbone="resnet18", pretrained=False, discrimator_level=1, input_size=100):
        super(ASPMTL, self).__init__()
        self.tasks = tasks
        if backbone == "resnet18":
            resnet = torchvision.models.resnet18(pretrained=pretrained)
        elif backbone == "resnet50":
            resnet = torchvision.models.resnet50(pretrained=pretrained)
        elif backbone == "resnet101":
            resnet = torchvision.models.resnet50(pretrained=pretrained)
        elif backbone == "resnet152":
            resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.subnets = torch.nn.ModuleDict({task:copy.deepcopy(resnet) for task in self.tasks})
        self.share_net = copy.deepcopy(resnet)
        self.avgpool = resnet.avgpool
        self.task_discrimator = task_discrimator(len(self.tasks), discrimator_level)

    def forward(self, inputs, beta):
        if not isinstance(inputs, dict):
            inputs = {task:inputs for task in self.tasks}
        x = {task:self.subnets[task].conv1(inputs[task]) for task in self.tasks}
        x = {task:self.subnets[task].bn1(x[task]) for task in self.tasks}
        x = {task:self.subnets[task].relu(x[task]) for task in self.tasks}
        x = {task:self.subnets[task].maxpool(x[task]) for task in self.tasks}
        x = {task:self.subnets[task].layer1(x[task]) for task in self.tasks}
        x = {task:self.subnets[task].layer2(x[task]) for task in self.tasks}   
        x = {task:self.subnets[task].layer3(x[task]) for task in self.tasks}
        x = {task:self.subnets[task].layer4(x[task]) for task in self.tasks}

        share_x = {task:self.share_net.conv1(inputs[task]) for task in self.tasks}
        share_x = {task:self.share_net.bn1(share_x[task]) for task in self.tasks}
        share_x = {task:self.share_net.relu(share_x[task]) for task in self.tasks}
        share_x = {task:self.share_net.maxpool(share_x[task]) for task in self.tasks}
        share_x = {task:self.share_net.layer1(share_x[task]) for task in self.tasks}
        share_x = {task:self.share_net.layer2(share_x[task]) for task in self.tasks}   
        share_x = {task:self.share_net.layer3(share_x[task]) for task in self.tasks}
        share_x = {task:self.share_net.layer4(share_x[task]) for task in self.tasks}

        subnes_out = {task:self.avgpool(x[task]).flatten(1) for task in self.tasks}
        share_out = {task:self.avgpool(share_x[task]).flatten(1) for task in self.tasks}
        dis_out = {task:self.task_discrimator(share_x[task]) for task in self.tasks}

        return subnes_out, share_out, dis_out

#===============================================================================
#PSMCNN
class PSMCNN(torch.nn.Module):
    def __init__(self, tasks, backbone="resnet18", pretrained=False, alpha_grain=1,input_size=100):
        super(PSMCNN, self).__init__()
        self.tasks = tasks
        if backbone == "resnet18":
            resnet = torchvision.models.resnet18(pretrained=pretrained)
        elif backbone == "resnet50":
            resnet = torchvision.models.resnet50(pretrained=pretrained)
        elif backbone == "resnet101":
            resnet = torchvision.models.resnet101(pretrained=pretrained)
        elif backbone == "resnet152":
            resnet = torchvision.models.resnet152(pretrained=pretrained)
        self.subnets = torch.nn.ModuleDict({task:copy.deepcopy(resnet) for task in self.tasks})
        self.share_net = copy.deepcopy(resnet)
        self.avgpool = resnet.avgpool

        if alpha_grain == 1:
            alpha = {task:torch.nn.Parameter(torch.tensor([0.5])).requires_grad_() for task in ['self', 'shared']}
            alpha_sha = {task:torch.nn.Parameter(torch.tensor([0.5])).requires_grad_() for task in self.tasks + ['shared']}
            layer_alpha = {task:copy.deepcopy(torch.nn.ParameterDict(alpha)) for task in tasks} # 浅拷贝？
            layer_alpha['shared'] = torch.nn.ParameterDict(alpha_sha)
            self.layer_alpha = torch.nn.ModuleList([torch.nn.ModuleDict(copy.deepcopy(layer_alpha)) for i in range(4)]) 

        elif alpha_grain == 2:
            temp_x = torch.randn((1,3,input_size, input_size))
            temp_x = resnet.conv1(temp_x)
            temp_x = resnet.maxpool(temp_x)

            temp_x = resnet.layer1(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in ['self', 'shared']}
            alpha_sha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in self.tasks + ['shared']}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer_alpha['shared'] = torch.nn.ParameterDict(alpha_sha)
            layer1_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            temp_x = resnet.layer2(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in ['self', 'shared']}
            alpha_sha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in self.tasks + ['shared']}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer_alpha['shared'] = torch.nn.ParameterDict(alpha_sha)
            layer2_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            temp_x = resnet.layer3(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in ['self', 'shared']}
            alpha_sha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in self.tasks + ['shared']}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer_alpha['shared'] = torch.nn.ParameterDict(alpha_sha)
            layer3_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            temp_x = resnet.layer4(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in ['self', 'shared']}
            alpha_sha =  {task:torch.nn.Parameter(torch.full((temp_x.size()[-1], temp_x.size()[-2]), 0.5)).requires_grad_() for task in self.tasks + ['shared']}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer_alpha['shared'] = torch.nn.ParameterDict(alpha_sha)
            layer4_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            self.layer_alpha = torch.nn.ModuleList([layer1_alpha, layer2_alpha, layer3_alpha, layer4_alpha])

        elif alpha_grain == 3:
            temp_x = torch.randn((1,3,input_size, input_size))
            temp_x = resnet.conv1(temp_x)
            temp_x = resnet.maxpool(temp_x)

            temp_x = resnet.layer1(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in ['self', 'shared']}
            alpha_sha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in self.tasks + ['shared']}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer_alpha['shared'] = torch.nn.ParameterDict(alpha_sha)
            layer1_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            temp_x = resnet.layer2(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in ['self', 'shared']}
            alpha_sha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in self.tasks + ['shared']}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer_alpha['shared'] = torch.nn.ParameterDict(alpha_sha)
            layer2_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            temp_x = resnet.layer3(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in ['self', 'shared']}
            alpha_sha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in self.tasks + ['shared']}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer_alpha['shared'] = torch.nn.ParameterDict(alpha_sha)
            layer3_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            temp_x = resnet.layer4(temp_x)
            alpha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in ['self', 'shared']}
            alpha_sha =  {task:torch.nn.Parameter(torch.full(temp_x.size()[1:], 0.5)).requires_grad_() for task in self.tasks + ['shared']}
            layer_alpha = {task:torch.nn.ParameterDict(alpha) for task in tasks}
            layer_alpha['shared'] = torch.nn.ParameterDict(alpha_sha)
            layer4_alpha = torch.nn.ModuleDict(copy.deepcopy(layer_alpha))

            self.layer_alpha = torch.nn.ModuleList([layer1_alpha, layer2_alpha, layer3_alpha, layer4_alpha])

    def forward(self, inputx):
        if not isinstance(inputx, dict):
            inputs = {task:inputx for task in self.tasks}
        x = {task:self.subnets[task].conv1(inputs[task]) for task in self.tasks}
        x = {task:self.subnets[task].bn1(x[task]) for task in self.tasks}
        x = {task:self.subnets[task].relu(x[task]) for task in self.tasks}
        x = {task:self.subnets[task].maxpool(x[task]) for task in self.tasks}

        share_x = self.share_net.conv1(inputx) 
        share_x = self.share_net.bn1(share_x) 
        share_x = self.share_net.relu(share_x) 
        share_x = self.share_net.maxpool(share_x) 

        x = {task:self.subnets[task].layer1(x[task]) for task in self.tasks}
        share_x = self.share_net.layer1(share_x)
        x, share_x = self._feature_fusion(x, share_x, self.layer_alpha[0])


        x = {task:self.subnets[task].layer2(x[task]) for task in self.tasks}
        share_x = self.share_net.layer2(share_x)
        x, share_x = self._feature_fusion(x, share_x, self.layer_alpha[1])


        x = {task:self.subnets[task].layer3(x[task]) for task in self.tasks}
        share_x = self.share_net.layer3(share_x)
        x, share_x = self._feature_fusion(x, share_x, self.layer_alpha[2])


        x = {task:self.subnets[task].layer4(x[task]) for task in self.tasks}
        share_x = self.share_net.layer4(share_x)
        x,_ = self._feature_fusion(x, share_x, self.layer_alpha[3])             


        x = {task:self.avgpool(x[task]) for task in self.tasks}
        subnet_out = {task:x[task].flatten(1) for task in self.tasks}

        return subnet_out
    def _feature_fusion(self, x, share_x, alpha):
        fused = {}
        temp = [share_x*alpha['shared']['shared']]
        for task in x.keys():
            fused[task] = x[task]*torch.sigmoid(alpha[task]['self']) + share_x*torch.sigmoid(alpha[task]['shared'])
            temp.append(x[task]*alpha['shared'][task])
        share_x = sum(temp) # 梯度断开？
        if len(x) == 1:
            task = list(x.keys())[0]
            return fused[task], share_x
        else:
            return fused, share_x 


if __name__ == "__main__":
    a = DMTL(['a','b'])
    print(a.layer1['a'].parameters())

